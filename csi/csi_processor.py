
import os
import numpy as np
import pandas as pd
import csi.loaders.huawei.csi_loader as csi_loader
import csi.csi_domain as csi_domain
from csi.loaders.huawei.csi_loader import CsiTruthDataLoader, CsiDataLoader
from utils.algo_utils import (
    get_doppler_spectrum_from_tensor,
    interpolate_csi,
    interpolate_rssi,
    get_scaled_csi_single
)


class CsiProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        all_spectrograms = []
        all_labels = []
        all_rssi_stats = []

        for datum in self.data:
            file_path = datum['file_path']
            data = datum['data']

            print(f"\n处理文件 {truth_path}: ")
            if len(data) == 0:
                print("  数据为空，跳过。")
                continue

            truth_path = f"{file_path[:-4]} _truth.txt"
            if not os.path.exists(truth_path):
                print(f"  无 {file_path}, 跳过")
                continue

            truth_labels = CsiTruthDataLoader.read_truth_file(truth_path)
            num_labels = len(truth_labels)
            print(f"  对应 truth 数目: {num_labels}")

            # 1) 提取时间戳 & 采样率
            timestamps, final_fs, uniform_time, num_rssi = self.extract_timestamps_and_frequency(
                data)

            # 2) 插值
            csi_uniform, rssi_uniform = self.interpolate_csi_and_rssi(
                data, timestamps, uniform_time, num_rssi)

            # 3) 缩放
            scaled_tensor = self.scale_csi_with_rssi(
                uniform_time, csi_uniform, rssi_uniform)

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

            t_frames, amp_spectrum, phase_spectrum = self.extract_amplitude_and_phase(
                doppler_complex)

            # 5) 按 2s窗口 (4帧) 切分
            self.split_and_label_spectrograms(all_spectrograms, all_labels, all_rssi_stats, truth_labels,
                                              num_labels, uniform_time, rssi_uniform, frame_ts, t_frames, amp_spectrum, phase_spectrum)

        print(
            f"\n[load_data_from_folder] 共收集 {len(all_spectrograms)} 个谱图, {len(all_labels)} 个标签.")

        # 不做“硬性” freq_dim 统一 => 返回可变
        # all_spectrograms: list of (2, freq_dim_i, 4), freq_dim_i 各不相同
        # all_rssi_stats: list of np.array, each shape=(rssi_dim_i,)
        # all_labels:     np.array of shape=(N,)
        return all_spectrograms, np.array(all_labels, dtype=np.int64), all_rssi_stats

    def split_and_label_spectrograms(self, all_spectrograms, all_labels, all_rssi_stats, truth_labels, num_labels, uniform_time, rssi_uniform, frame_ts, t_frames, amp_spectrum, phase_spectrum):
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

                # => (freq_dim_i, 4)
            sub_amp = amp_spectrum[0, :, start_f:end_f]
            # => (freq_dim_i, 4)
            sub_phase = phase_spectrum[0, :, start_f:end_f]
            # => (2, freq_dim_i, 4)
            multi_2d = np.stack([sub_amp, sub_phase], axis=0)

            all_spectrograms.append(multi_2d)
            all_labels.append(truth_labels[i])

            # RSSI
            mid_f = (start_f + end_f)//2
            mid_time = frame_ts[mid_f]
            idx_time = np.argmin(np.abs(uniform_time - mid_time))
            rssi4 = rssi_uniform[idx_time]  # => shape=(4,)
            all_rssi_stats.append(rssi4.copy())

    def extract_amplitude_and_phase(self, doppler_complex):
        freq_dim = doppler_complex.shape[1]
        t_frames = doppler_complex.shape[2]
        print(
            f"  doppler谱 shape={doppler_complex.shape} (freq_dim={freq_dim}, time_frames={t_frames})")

        # 幅度 + 相位
        # shape: (1, freq_dim, time_frames)
        amp_spectrum = np.abs(doppler_complex)
        # shape: (1, freq_dim, time_frames)
        phase_spectrum = np.angle(doppler_complex)
        return t_frames, amp_spectrum, phase_spectrum

    def scale_csi_with_rssi(self, uniform_time, csi_uniform, rssi_uniform):
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
        return scaled_tensor

    def interpolate_csi_and_rssi(self, data, timestamps, uniform_time, num_rssi):
        df = pd.DataFrame({'total_seconds': timestamps, 'data': data})
        csi_uniform = interpolate_csi(
            df, uniform_time, num_subcarriers=30, num_antennas=4)
        rssi_uniform = interpolate_rssi(data, uniform_time, num_rssi)
        print("  插值后 csi:", csi_uniform.shape, "rssi:", rssi_uniform.shape)
        return csi_uniform, rssi_uniform

    def extract_timestamps_and_frequency(self, data):
        timestamps = np.array([
            self._calc_seconds(d['ts'][0], d['ts'][1], d['ts'][2]) for d in data
        ])
        start_ts = timestamps.min()
        end_ts = timestamps.max()

        final_fs = data[0]['fs']  # 取第一个即可
        target_dt = 1.0 / final_fs
        uniform_time = np.arange(start_ts, end_ts, target_dt)
        num_rssi = len(data[0]['rssi'])
        print(
            f"  目标采样率: {final_fs} Hz, 均匀时间长度: {len(uniform_time)}, num_rssi ={num_rssi}")

        return timestamps, final_fs, uniform_time, num_rssi

    def _calc_seconds(self, hour, minute, second):
        """
        Convert hour, minute, and second to total seconds.
        """
        return hour * TimeConstants.SECONDS_IN_HOUR + minute * TimeConstants.SECONDS_IN_MINUTE + second


class TimeConstants:
    SECONDS_IN_HOUR = 3600
    SECONDS_IN_MINUTE = 60
