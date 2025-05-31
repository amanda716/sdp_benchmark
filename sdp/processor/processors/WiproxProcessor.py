import os
import random
from typing import List

import h5py
import numpy as np

from csi.csi_data import CSIData
from sdp.processor.base_processor import BaseProcessor


class WiproxProcessor(BaseProcessor):
    def process(self, data_list: List[CSIData], **kwargs):
        global file_path
        data_dict_list = []
        all_files = []
        folder_path = kwargs.get('folder_path', '')
        num_samples = kwargs.get('nums_samples', 100000)

        # 查找文件夹中的所有文件的路径
        for root, dirs, files in os.walk(folder_path):
            for fn in files:
                if fn.lower().endswith('.mat'):
                    all_files.append(os.path.join(root, fn))
        try:
            for file_path in all_files:
                # 使用 h5py 读取 .mat 文件 (v7.3)
                with h5py.File(file_path, 'r') as mat_file:
                    # 获取数据
                    iot_csi = mat_file['iot_csi'][:].T  # 550x1 cell
                    ue_csi = mat_file['ue_csi'][:].T  # 550x1 cell
                    dist_val = mat_file['dist_val'][:].T  # 550x1 cell
                    print(f"iot_csi shape: {iot_csi.shape}")
                    print(f"ue_csi shape: {ue_csi.shape}")
                    print(f"dist_val shape: {dist_val.shape}")

                    # 对每个场景提取实际数据
                    for i in range(len(iot_csi)):
                        iot_csi_data = iot_csi[i, 0]
                        iot_csi_data_arr = h5py.h5r.get_name(iot_csi_data, mat_file.id)
                        iot_csi_data_arr_val = mat_file[iot_csi_data_arr][:].T

                        ue_csi_data = ue_csi[i, 0]  # 每个 ue_csi[i] 是一个 cell
                        ue_csi_data_arr = h5py.h5r.get_name(ue_csi_data, mat_file.id)
                        ue_csi_data_arr_val = mat_file[ue_csi_data_arr][:].T

                        dist_val_data = dist_val[i, 0]  # 每个 dist_val[i] 是一个 cell
                        dist_val_data_arr = h5py.h5r.get_name(dist_val_data, mat_file.id)
                        dist_val_data_arr_val = mat_file[dist_val_data_arr][:].T

                        # 获取场景的 UE 数量（M）
                        M = ue_csi_data_arr_val.shape[0]

                        # 生成每个UE和每个IoT设备的配对
                        for m in range(M):  # 遍历每个UE
                            for n in range(6):  # 遍历每个IoT设备
                                # 生成输入对和标签
                                ue_csi_sample = ue_csi_data_arr_val[m, :, :]  # UE的CSI，维度[56, 9]
                                iot_csi_sample = iot_csi_data_arr_val[n, :, :]  # IoT的CSI，维度[56, 9]
                                dist_val_sample = dist_val_data_arr_val[m, n]  # 距离标签，标量

                                # Convert complex to float32 by stacking real and imaginary parts
                                # 将数据转化为包含实部和虚部的 2 通道输入数据
                                ue_csi_sample = np.stack((ue_csi_sample['real'], ue_csi_sample['imag']), axis=-1).astype(
                                    np.float32)
                                iot_csi_sample = np.stack((iot_csi_sample['real'], iot_csi_sample['imag']), axis=-1).astype(
                                    np.float32)

                                # 将数据保存为字典并添加到结果列表
                                data_dict = {
                                    'ue_csi_data': ue_csi_sample,
                                    'iot_csi_data': iot_csi_sample,
                                    'dist_val': dist_val_sample,
                                }
                                data_dict_list.append(data_dict)

                # Randomly select samples from data_dict_list
                num_samples_to_select = min(num_samples, len(data_dict_list))  # Select at most num_samples
                selected_data = random.sample(data_dict_list, num_samples_to_select)

                return selected_data  # Return the selected samples
        except Exception as e:
            raise ValueError(f"Error loading .mat file: {file_path}, {str(e)}")
