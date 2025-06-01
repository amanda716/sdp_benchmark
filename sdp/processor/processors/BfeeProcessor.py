import os
import re
from typing import List

import h5py
import numpy as np

from csi.csi_data import CSIData
from sdp.processor.base_processor import BaseProcessor
from sdp.reader.readers.bfee_reader import BfeeReader
from sdp.utils.bfee_utils import interpolate_csi, interpolate_rssi, scale_csi_block, compute_doppler_spectrum, \
    build_uniform_time, get_time_seconds


class BfeeProcessor(BaseProcessor):
    def process(self, data_list: List[CSIData], **kwargs):
        folder_path = kwargs.get('folder_path', '')
        task_type = kwargs.get('task_type', '')
        final_fs = kwargs.get('final_fs', 1000.0)

        dataset = self.load_data(data_list, folder_path, task_type)
        all_specs = []
        all_labels = []
        all_rssi = []

        if not dataset:
            print("[Error] No data loaded.")
            return all_specs, all_labels, all_rssi

        for item in dataset:
            try:
                fpath = item['file']
                label = item['label']  # Ensure the label is correctly extracted (handle HDF5 reference)

                # Check if label is an HDF5 reference, and dereference it
                if isinstance(label, h5py.Reference):
                    label = label[()]  # Dereference the object to get the actual value
                    print(f"Label (dereferenced): {label}")

                label = int(label) - 1  # Subtract 1 from the label if necessary
                recs = item['recs']
                N = len(recs)

                # Debugging: Check if we have the correct number of records
                print(f"File: {fpath}, Record count: {N}")
                if N < 2:
                    print(f"[Warning] Skipping file {fpath} due to insufficient data.")
                    continue

                bf_times = get_time_seconds(recs, 'us')
                if bf_times.size == 0:
                    print(f"[Warning] Skipping {fpath} due to missing times.")
                    continue

                t0, t1 = bf_times[0], bf_times[-1]
                uniform_time = build_uniform_time(t0, t1, final_fs)

                Nrx = recs[0].n_rx
                Ntx = recs[0].n_tx
                csi_intp = interpolate_csi(recs, bf_times, uniform_time, Nrx=Nrx, Ntx=Ntx)
                rssi_intp = interpolate_rssi(recs, bf_times, uniform_time)

                scaled_arr = np.zeros_like(csi_intp, dtype=np.complex128)
                T_ = len(uniform_time)
                for i in range(T_):
                    block = csi_intp[i]
                    rssi3 = rssi_intp[i]
                    scaled = scale_csi_block(block, rssi3, noise_db=-92, Ntx=Ntx, Nrx=Nrx)
                    scaled_arr[i] = scaled

                spec_2d, freq_sel, t_sel = compute_doppler_spectrum(scaled_arr, rssi_intp, fs=final_fs)

                if spec_2d.shape[1] == 0 or spec_2d.shape[2] == 0:
                    print(f"[Warning] Got empty spectrum for {fpath}.")
                    continue

                all_specs.append(spec_2d)
                all_labels.append(label)
                all_rssi.append(rssi_intp[-1].copy())

            except Exception as e:
                print(f"[Error] Failed to process file {item['file']}: {e}")
                continue

        return all_specs, np.array(all_labels, dtype=int), all_rssi

    def load_data(self, data_list: List[CSIData], folder_path, task_type):
        """
        递归查找 .dat 文件，自动处理不同任务类型的文件。
        """
        all_files = []
        results = []

        for item in data_list:
            file = item.file_name
            recs = item.frames
            parsed_info = self.parse_file_info_from_filename(file, task_type)
            if parsed_info is None:
                print(f"Skipping file {file}, invalid format or filename does not match expected pattern.")
                continue
            user_id, gesture_type, torso_position, orientation, data_serial, receiver_number = parsed_info

            results.append({
                'file': file,
                'label': user_id,
                'gesture_type': gesture_type,
                'recs': recs
            })
            print(f"[Info] Loaded {file} with {len(recs)} records.")

        print(f"[Done] Found {len(results)} valid files in {folder_path}.")
        return results

    def parse_file_info_from_filename(self, f_name, task_type):
        """
        Parse the file name and return relevant information based on task_type.
        For Gesture Recognition: "id-a-b-c-d-Rx.dat"
        For Gait Recognition: "user1-1-1-r1.dat" => "user1"
        """
        base = os.path.splitext(os.path.basename(f_name))[0]

        if task_type == 'Gesture Recognition':
            m = re.match(r'user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)', base)
            if m:
                user_id = int(m.group(1))
                gesture_type = int(m.group(2))
                torso_position = int(m.group(3))
                orientation = int(m.group(4))
                data_serial = int(m.group(5))
                receiver_number = int(m.group(6))
                return user_id, gesture_type, torso_position, orientation, data_serial, receiver_number
            else:
                print(f"[Warning] Skipping file {f_name}: Invalid format for Gesture Recognition.")

        elif task_type == 'Activity Recognition':
            # Parse for Gait Recognition (pattern "user3-1-1-1-1-r1.dat")
            m = re.search(r'user(\d+)', base, re.IGNORECASE)
            if m:
                user_id = int(m.group(1))
                return user_id, None, None, None, None, None
            else:
                print(f"[Warning] Skipping file {f_name}: Invalid format for Activity Recognition.")

        else:
            print(f"[Error] Unknown task type: {task_type}")
