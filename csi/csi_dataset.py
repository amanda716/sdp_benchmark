
import torch
from torch.utils.data import Dataset
import os


class CSIDataset(Dataset):
    """
    A dataset class for loading and processing the CSI dataset.
    """

    def __init__(self, spectrograms, labels, rssi_stats):
        """
        Initializes the CSIDataset with spectrograms, labels, and RSSI statistics.

        Args:
            spectrograms (list): List of spectrograms.
            labels (list): List of labels corresponding to the spectrograms.
            rssi_stats (list): List of RSSI statistics.
        """
        self.spectrograms = spectrograms
        self.labels = labels
        self.rssi_stats = rssi_stats

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.spectrograms)
    
    def __getitem__(self, idx):
        doppler = self.spectrograms[idx]
        label = self.labels[idx]
        rssi = self.rssi_stats[idx]
        
        doppler_tensor = torch.from_numpy(doppler)
        rssi_tensor = torch.from_numpy(rssi)
        return doppler_tensor, rssi_tensor, label
