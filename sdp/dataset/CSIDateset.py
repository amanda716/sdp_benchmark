import torch
from torch.utils.data import Dataset


class CSIDataset(Dataset):

    def __init__(self, csi_data, labels, task_type, transform=None):
        """
        初始化CSI数据集

        参数:
            csi_data: 预处理后的CSI数据列表 (每个元素是5D数组)
            labels: 标签数组
            task_type: 任务类型 ('classification' 或 'regression')
            transform: 可选的额外变换
        """
        self.csi_data = csi_data
        self.labels = labels
        self.task_type = task_type
        self.transform = transform

        if len(csi_data) != len(labels):
            raise ValueError("CSI data and labels must have the same length")

    def __len__(self):
        return len(self.csi_data)

    def __getitem__(self, idx):
        csi_sample = self.csi_data[idx]
        csi_tensor = torch.tensor(csi_sample, dtype=torch.float32)
        label = self.labels[idx]

        # 根据任务类型转换标签类型
        if self.task_type == 'classification':
            label = torch.tensor(label, dtype=torch.long)
        else:  # regression
            label = torch.tensor(label, dtype=torch.float32)

        return csi_tensor, label


def csi_collate_fn(batch):
    """自定义collate函数处理可变尺寸的CSI数据"""
    csi_samples, labels = zip(*batch)

    max_time = max(s.shape[0] for s in csi_samples)
    max_freq = max(s.shape[1] for s in csi_samples)
    max_rx = max(s.shape[2] for s in csi_samples)
    max_tx = max(s.shape[3] for s in csi_samples)

    padded_csi = torch.zeros(
        len(csi_samples),
        max_time,
        max_freq,
        max_rx,
        max_tx,
        csi_samples[0].shape[-1]  # 通道维度 (实部和虚部)
    )

    for i, csi in enumerate(csi_samples):
        time, freq, rx, tx, channels = csi.shape
        padded_csi[i, :time, :freq, :rx, :tx, :] = csi

    labels = torch.stack(labels)
    return padded_csi, labels
