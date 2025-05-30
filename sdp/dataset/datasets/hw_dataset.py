from csi.csi_data import CSIData
from sdp.dataset.base_dataset import BaseDataset


class HwDataset(BaseDataset):
    def __init__(self, all_spectrograms, all_labels, all_rssi_stats):
        super().__init__()
