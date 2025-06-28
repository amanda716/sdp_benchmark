import torch
from torch import nn


class HwWidarModel(nn.Module):
    def __init__(self,
                 task_type='classification',
                 doppler_in_channels=2,  # (2=幅度+相位)
                 out_freq_dim=128,  # 自适应池化后 freq_dim
                 time_dim=4,  # time_dim 固定为4
                 doppler_hidden_size=64,  # LSTM hidden size for Doppler
                 max_rssi_dim=4,  # 最大 RSSI 维度
                 rssi_proj_dim=8,  # 投影后的 RSSI 维度
                 rssi_hidden_size=32,  # LSTM hidden size for RSSI
                 conv_channels=16,  # 卷积输出通道数
                 fix_n=128,  # 自适应池化后的freq_dim_widar
                 num_classes=4):
        super(HwWidarModel, self).__init__()
        self.task_type = task_type
        self.out_freq_dim = out_freq_dim
        self.time_dim = time_dim
        self.doppler_in_channels = doppler_in_channels

        # ========== 1) Doppler 分支: 参考widar的CNN + 全连接 ==========
        self.freq_pool = nn.AdaptiveMaxPool1d(fix_n)
        # 3D卷积层
        self.conv1_widar = nn.Conv3d(
            in_channels=2,  # 幅度和相位双通道
            out_channels=conv_channels,
            kernel_size=(3, 3, 1),  # (频率核, 时间核, 虚拟维度)
            padding=(1, 1, 0)  # 保持频率和时间维度尺寸
        )
        self.conv_channels = conv_channels
        self.pool_widar = nn.MaxPool3d(kernel_size=(2, 2, 1))
        self.adaptive_pool_widar = nn.AdaptiveAvgPool3d((4, 2, 1))
        self.flatten = nn.Flatten()
        self.fc1_widar = nn.Linear(conv_channels * time_dim * doppler_in_channels * 1,
                                   256)  # 输入尺寸计算: 16通道 * 4频率 * (幅值+相位通道) * 1虚拟维度
        self.fc2_widar = nn.Linear(256, 128)
        self.dropout_widar = nn.Dropout(0.5)

        # ========== 2) RSSI 分支: 投影 + LSTM ==========
        self.rssi_proj = RSSIProjector(max_rssi_dim=max_rssi_dim, proj_dim=rssi_proj_dim)
        self.rssi_lstm = nn.LSTM(
            input_size=rssi_proj_dim,
            hidden_size=rssi_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout_rssi = nn.Dropout(p=0.3)  # 手动加

        # ========== 3) 融合全连接层 ==========
        fusion_dim = 128 + rssi_hidden_size  # doppler_size + rssi_hidden_size

        # 根据任务类型设置输出层
        if self.task_type == 'classification':
            self.fc_fusion = nn.Linear(fusion_dim, num_classes)
        elif self.task_type == 'regression':
            self.fc_fusion = nn.Linear(fusion_dim, 1)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def forward(self, doppler_list, rssi_list):
        """
        doppler_list: list of B tensors, each shape=(2, freq_dim_i, 4)
        rssi_list:   list of B tensors, each shape=(rssi_dim_i,)

        返回: logits: (B, num_classes)
        """
        B = len(doppler_list)

        # 1) 处理 RSSI 分支
        rssi_proj_batch = []
        for i in range(B):
            rssi = rssi_list[i].to(next(self.parameters()).device)
            # 先投影 => (proj_dim,)
            r_proj = self.rssi_proj(rssi)  # => shape=(proj_dim,)
            # => (1, seq_len=1, proj_dim)
            r_proj = r_proj.unsqueeze(0).unsqueeze(0)
            out_r, _ = self.rssi_lstm(r_proj)  # => (1,1,rssi_hidden_size)
            hidden_rssi = out_r[:, -1, :]  # =>(1,rssi_hidden_size)
            hidden_rssi = self.dropout_rssi(hidden_rssi)  # dropout
            rssi_proj_batch.append(hidden_rssi)

        # 拼接成 (B, rssi_hidden_size)
        rssi_batch = torch.cat(rssi_proj_batch, dim=0)

        # 2) 处理 Doppler 分支
        doppler_hidden_list = []
        for i in range(B):
            x = doppler_list[i].to(next(self.parameters()).device)  # shape=(2,freq_dim_i, 4)
            x = x.permute(0, 2, 1)  # (2, 4, freq_dim_i)
            x = self.freq_pool(x)  # (2, 4, fix_n)
            x = x.permute(0, 2, 1)  # (2, fix_n, 4)
            x = x.unsqueeze(-1)  # (2, fix_n, 4, 1)'
            doppler_hidden_list.append(x)  # (B, 2, fix_n, 4, 1)
            # stack element in list to create batches
        doppler_hidden_batch = torch.stack(doppler_hidden_list)
        # CNN_widar
        s = nn.functional.relu(self.conv1_widar(doppler_hidden_batch))  # (B, 16, fix_n, 4, 1)
        s = self.pool_widar(s)  # (B, 16, fix_n//2, 2, 1)
        s = self.adaptive_pool_widar(s)  # (B, 16, 4, 2, 1)
        # 全连接特征压缩
        s = self.flatten(s)  # (B, 16*4*2*1=128)
        s = nn.functional.relu(self.fc1_widar(s))  # (B, 256)
        s = self.dropout_widar(s)
        s = nn.functional.relu(self.fc2_widar(s))  # (B, 128)
        s = self.dropout_widar(s)

        # 3) 融合
        fusion = torch.cat([s, rssi_batch], dim=1)  # =>(B, 128 + rssi_hidden_size)
        logits = self.fc_fusion(fusion)  # =>(B, num_classes)

        # 根据任务类型选择输出方式
        if self.task_type == 'classification':
            return logits
        elif self.task_type == 'regression':
            return logits.squeeze()  # 对于回归任务，去除维度


class RSSIProjector(nn.Module):
    """
    演示用的 RSSI 投影层: 把可变维度的 rssi(<=max_rssi_dim) 填充/截断到 max_rssi_dim，
    再用一个全连接投影到固定 rssi_proj_dim 维。
    """

    def __init__(self, max_rssi_dim=4, proj_dim=8):
        super().__init__()
        self.max_rssi_dim = max_rssi_dim
        self.proj = nn.Linear(max_rssi_dim, proj_dim)

    def forward(self, rssi_tensor):
        # rssi_tensor: shape=(rssi_dim_i,)  可能 < max_rssi_dim or =max_rssi_dim
        rssi_dim = rssi_tensor.shape[0]
        # 1) 若 rssi_dim < max_rssi_dim => padding
        #    若 rssi_dim > max_rssi_dim => 截断
        pad_rssi = torch.zeros(self.max_rssi_dim, device=rssi_tensor.device, dtype=rssi_tensor.dtype)
        if rssi_dim >= self.max_rssi_dim:
            pad_rssi[:] = rssi_tensor[:self.max_rssi_dim]
        else:
            pad_rssi[:rssi_dim] = rssi_tensor

        # 2) 全连接投影 => (proj_dim,)
        out = self.proj(pad_rssi)  # => shape=(proj_dim,)
        return out
