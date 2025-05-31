import torch

from sdp.dataset.datasets.bfee_dataset import BfeeDataset


class BFEEPositionTrackingDataset(BfeeDataset):
    def __init__(self, data):
        """
        Custom Dataset for Positioning and Tracking using Wi-Prox data.
        """
        super().__init__()
        self.data = data
        self.valid_data = []

        # Filter out invalid samples (those without the required keys)
        for sample in self.data:
            if 'iot_csi_data' in sample and 'ue_csi_data' in sample and 'dist_val' in sample:
                self.valid_data.append(sample)

        print(f"Filtered valid data size: {len(self.valid_data)}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        sample = self.valid_data[idx]

        # Extract the CSI data (IoT and UE) and distance values (labels)
        iot_csi = sample['iot_csi_data']
        ue_csi = sample['ue_csi_data']
        dist_val = sample['dist_val']

        # Convert to tensors
        iot_csi_tensor = torch.tensor(iot_csi, dtype=torch.float32)
        ue_csi_tensor = torch.tensor(ue_csi, dtype=torch.float32)
        dist_val_tensor = torch.tensor(dist_val, dtype=torch.float32)

        return iot_csi_tensor, ue_csi_tensor, dist_val_tensor
