import torch

from sdp.dataset.datasets.bfee_dataset import BfeeDataset


class GenDataset(BfeeDataset):
    def __init__(self, specs, labels, rssi, gesture_types=None):
        super().__init__()
        self.specs = specs
        self.labels = labels
        self.rssi = rssi
        self.gesture_types = gesture_types

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec_i = self.specs[idx]  # (2, freq_dim_i, time_frames)
        lab_i = self.labels[idx]  # User ID
        rssi_i = self.rssi[idx]  # (3,)
        if self.gesture_types is not None:
            # Gesture type for gesture recognition
            gesture_type_i = self.gesture_types[idx]
            return spec_i, rssi_i, lab_i, gesture_type_i
        return spec_i, rssi_i, lab_i

    def doppler_collate_fn(batch):
        global spec_np, rssi_np, lab
        specs_list = []
        rssi_list = []
        labels_list = []
        gesture_types_list = []

        # Check if the batch contains 4 elements (including gesture_type)
        for item in batch:
            if len(item) == 4:
                spec_np, rssi_np, lab, gesture_type = item
                gesture_types_list.append(gesture_type)
            elif len(item) == 3:
                spec_np, rssi_np, lab = item
                gesture_types_list.append(None)  # If no gesture type, append None

            specs_list.append(torch.from_numpy(spec_np).float())
            rssi_list.append(torch.from_numpy(rssi_np).float())
            labels_list.append(lab)

        # Convert lists to tensors
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        gesture_types_tensor = torch.tensor(gesture_types_list, dtype=torch.long) if gesture_types_list[
                                                                                         0] is not None else None

        # Return the tensors
        if gesture_types_tensor is not None:
            return specs_list, rssi_list, labels_tensor, gesture_types_tensor
        else:
            return specs_list, rssi_list, labels_tensor  # If no gesture_type, return only 3 items
