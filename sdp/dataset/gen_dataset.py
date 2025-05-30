from torch.utils.data import Dataset


class GenDataset(Dataset):
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
