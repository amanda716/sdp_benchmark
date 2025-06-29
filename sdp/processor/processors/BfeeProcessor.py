import os
import re
from typing import List

import numpy as np
import torch

from csi.csi_data import CSIData
from sdp.processor.base_processor import BaseProcessor


class BfeeProcessor(BaseProcessor):
    def process(self, data_list: List[CSIData], **kwargs):
        sub_type = kwargs.get('sub_type', '')

        all_tensors = []
        all_labels = []
        # one CSIData -> one tensor: [time, sub_carrier, tx, rx]
        for csi_data in data_list:
            # get label(user_id or gesture_type ?)  current for gesture_type: parsed_info[1]
            parsed_info = self.parse_file_info_from_filename(csi_data.file_name, sub_type)
            print(f"parsed_info: {parsed_info}")
            print(f"len of csi_data.frames: {len(csi_data.frames)}")
            all_labels.extend([int(parsed_info[1]) - 1] * len(csi_data.frames))

            # generate four_dimension tensor
            if not csi_data.frames:
                all_tensors.append(torch.tensor([]))
                continue
            temp_list = []
            for frame in csi_data.frames:
                csi_tensor = torch.from_numpy(frame.csi_array.astype(np.complex64))
                # [a, b, rx, tx] -> [a, b, tx, rx]
                csi_permuted = csi_tensor.permute(0, 2, 1)
                temp_list.append(csi_permuted)
                file_tensor = torch.stack(temp_list, dim=0)
                all_tensors.append(file_tensor)
        return all_tensors, all_labels

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
