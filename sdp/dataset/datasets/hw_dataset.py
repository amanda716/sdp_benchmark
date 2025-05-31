import torch
from sdp.dataset.base_dataset import BaseDataset


class HwDataset(BaseDataset):
    def __init__(self, all_spectrograms, all_labels, all_rssi_stats):
        """
        all_spectrograms: list of np.array, each shape=(2, freq_dim_i, 4)
        all_labels: np.array of shape=(N,)
        all_rssi_stats: list of np.array, each shape=(rssi_dim_i,)
        """
        super().__init__()
        self.all_spectrograms = all_spectrograms
        self.all_labels = all_labels
        self.all_rssi_stats = all_rssi_stats
        assert len(self.all_spectrograms) == len(self.all_labels) == len(self.all_rssi_stats)

    def __len__(self):
        return len(self.all_spectrograms)

    def __getitem__(self, idx):
        doppler_2d = self.all_spectrograms[idx]  # shape=(2, freq_dim_i, 4), numpy
        label = self.all_labels[idx]
        rssi_arr = self.all_rssi_stats[idx]  # shape=(rssi_dim_i,), list

        # 转换为 torch tensor
        doppler_tensor = torch.from_numpy(doppler_2d).float()  # (2, freq_dim_i, 4)
        rssi_tensor = torch.from_numpy(rssi_arr).float()  # (rssi_dim_i,)

        return doppler_tensor, rssi_tensor, label

    def doppler_collate_fn(batch):
        """
        batch: list of tuples (doppler_tensor, rssi_tensor, label)
            doppler_tensor: (2, freq_dim_i, 4)
            rssi_tensor: (rssi_dim_i,)
            label: int
        返回:
            doppler_list: list of torch.Tensor, each shape=(2, freq_dim_i, 4)
            rssi_list: list of torch.Tensor, each shape=(rssi_dim_i,)
            label_tensor: torch.Tensor of shape=(B,)
        """
        doppler_list = []
        rssi_list = []
        label_list = []
        for (dopp_t, rssi_t, lab) in batch:
            doppler_list.append(dopp_t)  # (2, freq_dim_i, 4)
            rssi_list.append(rssi_t)  # (rssi_dim_i,)
            label_list.append(lab)  # int
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        return doppler_list, rssi_list, label_tensor
