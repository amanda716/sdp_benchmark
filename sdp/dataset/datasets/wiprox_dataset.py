import torch
from torch.utils.data import Dataset


class WiProxDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        ue_csi = torch.tensor(sample['ue_csi_data'], dtype=torch.float32)
        iot_csi = torch.tensor(sample['iot_csi_data'], dtype=torch.float32)
        dist_val = torch.tensor(sample['dist_val'], dtype=torch.float32)
        # Reshape here for MRSE
        ue_csi = ue_csi.permute(2, 0, 1)  # Reshape for MRSE
        iot_csi = iot_csi.permute(2, 0, 1)  # Reshape for MRSE
        return ue_csi, iot_csi, dist_val
