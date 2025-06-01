import os
import random
import numpy as np
from typing import List

import pandas as pd

from csi.csi_data import CSIData
from sdp.processor.base_processor import BaseProcessor
from sdp.utils.algo_utils import interpolate_csi, interpolate_rssi, get_scaled_csi_single, \
    get_doppler_spectrum_from_tensor


def read_truth_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().strip().split()
    labels = [int(x) for x in lines]
    return labels


class HwProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def process(self, data_list: List[CSIData], **kwargs):
        """
        hw process flow
        :param data_list:
        :param kwargs: for folder_path
        :return:
        """
        folder_path = kwargs.get('folder_path', '')
        # 确保有数据可选
        if data_list:
            random_record = random.choice(data_list)  # 从列表中随机选取一个文件
            file_name = random_record.file_name
            data = random_record.frames
            print(len(data))

            print(f"\n随机选取的数据来自文件: {file_name}")

            # 确保数据列表有足够的条目
            if len(data) >= 2:
                first_packet = data[0]  # 第一条数据
                second_packet = data[1]  # 第二条数据
                last_packet = data[-1]  # 最后一条数据

                def print_packet(packet, index):
                    print(f"\n=== 第 {index} 条 CSI 数据包 ===")
                    print(f"Timestamp (ts): {packet.ts}")
                    print(f"RSSI: {packet.rssi}")
                    print(f"MCS: {packet.mcs}")
                    print(f"Gain: {packet.gain}")
                    print("CSI (full data):")
                    # print(packet.csi_matrix)  # 完整展示CSI数据
                    # print(f"Sampling Rate (fs): {packet['fs']}")

                print_packet(first_packet, "1")
                print_packet(second_packet, "2")
                print_packet(last_packet, "最后")
            else:
                print(f"文件 {file_name} 记录数不足 2 条，无法打印完整示例。")
        else:
            print("没有数据文件可选。")

        print(f"[load_data_from_folder] 读取 {len(data_list)} 个文件于 {folder_path}")

        all_spectrograms = []
        all_labels = []
        all_rssi_stats = []

        for item in data_list:
            file_name = item.file_name
            data = item.frames
            prefix = file_name[:-4]
            truth_file = prefix + "_truth.txt"
            print(f"\nHwProcessor 处理文件 {file_name}: ")
            if len(data) == 0:
                print("  数据为空，跳过。")
                continue

            truth_path = ""
            file_generator = (
                os.path.join(root, file)
                for root, dirs, files in os.walk(folder_path)
                for file in files
                if file == truth_file
            )
            try:
                match = next(file_generator)
                truth_path = match
                print(f"truth_path:{truth_path}")
                truth_labels = read_truth_txt(truth_path)
                num_labels = len(truth_labels)
                print(f"  对应 truth 数目: {num_labels}")
            except Exception as e:
                print(f"not found exception: {e}")
                print(f"  无 {truth_file}, 跳过")
                continue

            # 1) 提取时间戳 & 采样率
            timestamps = np.array([
                rec.ts[0] * 3600 + rec.ts[1] * 60 + rec.ts[2] for rec in data
            ])
            start_time = timestamps.min()
            end_time = timestamps.max()

            final_fs = data[0].fs  # 取第一个即可
            target_dt = 1.0 / final_fs
            uniform_time = np.arange(start_time, end_time, target_dt)
            num_rssi = len(data[0].rssi)
            print(f"  目标采样率: {final_fs} Hz, 均匀时间长度: {len(uniform_time)}, num_rssi ={num_rssi}")

            # 2) 插值
            df_temp = pd.DataFrame({'total_seconds': timestamps, 'data': data})
            csi_uniform = interpolate_csi(df_temp, uniform_time, num_subcarriers=30, num_antennas=4)
            rssi_uniform = interpolate_rssi(data, uniform_time, num_rssi)
            print("  插值后 csi:", csi_uniform.shape, "rssi:", rssi_uniform.shape)

            # 3) 缩放
            scaled_tensor = np.zeros_like(csi_uniform, dtype=np.complex128)
            for t in range(len(uniform_time)):
                csi_val = csi_uniform[t]
                rssi_val = rssi_uniform[t]
                scaled_csi = get_scaled_csi_single(
                    csi_val, rssi_val, noise_db=-92, Ntx=1, Nrx=4
                )
                # 这里假设已统一截取前30个子载波
                scaled_tensor[t] = scaled_csi[:30, :4]
            print(f"  scaled_tensor shape={scaled_tensor.shape}")

            # 4) 生成多普勒谱 (复数)
            doppler_complex, freq_bin, frame_ts = get_doppler_spectrum_from_tensor(
                scaled_tensor, timestamps,
                rx_cnt=1, rx_acnt=4,
                method='stft',
                fs=final_fs,
                window_size_seconds=2.0,
                hop_size_seconds=0.5
            )
            if doppler_complex.shape[2] == 0:
                print("  多普勒谱为空，跳过。")
                continue

            freq_dim = doppler_complex.shape[1]
            t_frames = doppler_complex.shape[2]
            print(f"  doppler谱 shape={doppler_complex.shape} (freq_dim={freq_dim}, time_frames={t_frames})")

            # 幅度 + 相位
            amp_spectrum = np.abs(doppler_complex)  # shape: (1, freq_dim, time_frames)
            phase_spectrum = np.angle(doppler_complex)  # shape: (1, freq_dim, time_frames)

            # 5) 按 2s窗口 (4帧) 切分
            window_size_seconds = 2.0
            hop_size_seconds = 1.5

            frames_per_window = int(window_size_seconds / 0.5)  #
            number_of_windows = (t_frames - 1) // frames_per_window + 1

            # 让 number_of_windows 跟 label 数量对齐
            if number_of_windows > num_labels:
                number_of_windows = num_labels
            elif number_of_windows < num_labels:
                # 说明 label 多余 => 截断 label
                truth_labels = truth_labels[:number_of_windows]

            for i in range(number_of_windows):
                start_f = i * frames_per_window
                end_f = start_f + frames_per_window
                if end_f > t_frames:
                    break

                sub_amp = amp_spectrum[0, :, start_f:end_f]  # => (freq_dim_i, 4)
                sub_phase = phase_spectrum[0, :, start_f:end_f]  # => (freq_dim_i, 4)
                multi_2d = np.stack([sub_amp, sub_phase], axis=0)  # => (2, freq_dim_i, 4)

                all_spectrograms.append(multi_2d)
                all_labels.append(truth_labels[i])

                # RSSI
                mid_f = (start_f + end_f) // 2
                mid_time = frame_ts[mid_f]
                idx_time = np.argmin(np.abs(uniform_time - mid_time))
                rssi4 = rssi_uniform[idx_time]  # => shape=(4,)
                all_rssi_stats.append(rssi4.copy())

        print(f"\n[load_data_from_folder] 共收集 {len(all_spectrograms)} 个谱图, {len(all_labels)} 个标签.")

        # 不做“硬性” freq_dim 统一 => 返回可变
        # all_spectrograms: list of (2, freq_dim_i, 4), freq_dim_i 各不相同
        # all_rssi_stats: list of np.array, each shape=(rssi_dim_i,)
        # all_labels:     np.array of shape=(N,)
        return all_spectrograms, np.array(all_labels, dtype=np.int64), all_rssi_stats
