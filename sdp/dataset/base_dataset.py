from torch.utils.data import Dataset
from csi.csi_data import CSIData
import torch


class BaseDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None
