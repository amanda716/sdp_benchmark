import torch
import torch.nn as nn


class RSSIProjector(nn.Module):
    """
    RSSIProjector 类旨在处理 RSSI（接收信号强度指示）输入的可变维度，并将其投影到固定维度的表示。
    """

    def __init__(self, max_rssi_dim=4, proj_dim=8):
        super().__init__()
        self.max_rssi_dim = max_rssi_dim
        self.projection_layer = nn.Linear(max_rssi_dim, proj_dim)

    def forward(self, tensor):
        """
        前向传播方法，处理输入的 RSSI 张量。

        参数:
            tensor (torch.Tensor): 输入的 RSSI 张量，形状为 (batch_size, max_rssi_dim)。

        返回:
            torch.Tensor: 投影后的张量，形状为 (batch_size, proj_dim)。
        """
        dim = tensor.shape[0]
        padded_tensor = torch.zeros(
            self.max_rssi_dim,  device=tensor.device, dtype=tensor.dtype)
        if dim > self.max_rssi_dim:
            padded_tensor[:] = tensor[:self.max_rssi_dim]
        else:
            padded_tensor[:dim] = tensor
        return self.projection_layer(padded_tensor)


class DopplerRSSIFusionModel(nn.Module):
    def __init__(self,
                 task_type='classification',
                 doppler_in_channels=2,  # (2=幅度+相位)
                 out_freq_dim=128,       # 自适应池化后 freq_dim
                 time_dim=4,             # time_dim 固定为4
                 doppler_hidden_size=64,  # LSTM hidden size for Doppler
                 rssi_max_dim=4,         # 最大 RSSI 维度
                 rssi_proj_dim=8,        # 投影后的 RSSI 维度
                 rssi_hidden_size=32,    # LSTM hidden size for RSSI
                 num_classes=4):
        super(DopplerRSSIFusionModel, self).__init__()
        self.task_type = task_type
        self.out_freq_dim = out_freq_dim
        self.time_dim = time_dim
        self.doppler_in_channels = doppler_in_channels

        # 1) Doppler 分支: CNN + Adaptive Pooling + MaxPool + LSTM
        self.conv1 = nn.Conv2d(in_channels=doppler_in_channels,
                               out_channels=8,
                               kernel_size=3,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_freq_dim, time_dim))
        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16,
                               kernel_size=3,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 可以在 CNN 输出后/前插入 dropout
        self.dropout_cnn = nn.Dropout(p=0.3)

        # LSTM for Doppler
        # 如果仅 num_layers=1，LSTM内置的dropout并不会生效
        # 下面演示在 LSTM 输出后手动加 dropout
        self.doppler_lstm = nn.LSTM(
            input_size=16 * (out_freq_dim // 2),
            hidden_size=doppler_hidden_size,
            num_layers=1,
            batch_first=True
            # dropout=0.3, # 只有 num_layers>1 时生效
        )
        # 手动在 LSTM 输出后添加 dropout
        self.dropout_lstm = nn.Dropout(p=0.3)

        # 2) RSSI 分支: 投影 + LSTM
        self.rssi_proj = RSSIProjector(
            max_rssi_dim=rssi_max_dim, proj_dim=rssi_proj_dim)
        self.rssi_lstm = nn.LSTM(
            input_size=rssi_proj_dim,
            hidden_size=rssi_hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.dropout_rssi = nn.Dropout(p=0.3)  # 手动加

        # 3) 融合全连接层
        fusion_dim = doppler_hidden_size + rssi_hidden_size

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

        返回: logits: (sz, num_classes)
        """
        sz = len(doppler_list)

        # 1) 处理 RSSI 分支
        rssi_proj_batch = []
        for i in range(sz):
            rssi = rssi_list[i].to(next(self.parameters()).device)
            # 先投影 => (proj_dim,)
            r_proj = self.rssi_proj(rssi)      # => shape=(proj_dim,)
            # => (1, seq_len=1, proj_dim)
            r_proj = r_proj.unsqueeze(0).unsqueeze(0)
            out_r, _ = self.rssi_lstm(r_proj)  # => (1,1,rssi_hidden_size)
            hidden_rssi = out_r[:, -1, :]      # =>(1,rssi_hidden_size)
            hidden_rssi = self.dropout_rssi(hidden_rssi)  # dropout
            rssi_proj_batch.append(hidden_rssi)

        # 拼接成 (sz, rssi_hidden_size)
        rssi_batch = torch.cat(rssi_proj_batch, dim=0)

        # 2) 处理 Doppler 分支
        doppler_hidden_batch = []
        for i in range(sz):
            # shape=(2,freq_dim_i, 4)
            x = doppler_list[i].to(next(self.parameters()).device)
            x = x.unsqueeze(0)  # =>(1,2,freq_dim_i,4)
            # CNN
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.adaptive_pool(x)    # =>(1,8,out_freq_dim, time_dim)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.pool(x)             # =>(1,16,out_freq_dim//2,time_dim//2)

            x = self.dropout_cnn(x)      # CNN后再 dropout

            # Reshape => LSTM
            # x.shape=(1,16,out_freq_dim//2, time_dim//2)
            _, C, Fh, Tw = x.shape
            # =>(1, 16*(out_freq_dim//2), time_dim//2)
            x = x.view(1, C*Fh, Tw)
            # =>(1, time_dim//2, 16*(out_freq_dim//2))
            x = x.permute(0, 2, 1)

            # =>(1, time_dim//2, doppler_hidden_size)
            out_d, _ = self.doppler_lstm(x)
            hidden_doppler = out_d[:, -1, :]  # =>(1, doppler_hidden_size)
            hidden_doppler = self.dropout_lstm(hidden_doppler)
            doppler_hidden_batch.append(hidden_doppler)

        # =>(sz, doppler_hidden_size)
        doppler_batch = torch.cat(doppler_hidden_batch, dim=0)

        # 3) 融合
        # =>(sz, doppler_hidden_size + rssi_hidden_size)
        fusion = torch.cat([doppler_batch, rssi_batch], dim=1)
        # =>(sz, num_classes)
        logits = self.fc_fusion(fusion)

        # 根据任务类型选择输出方式
        if self.task_type == 'classification':
            return logits
        elif self.task_type == 'regression':
            return logits.squeeze()  # 对于回归任务，去除维度
