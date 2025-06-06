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
                 num_classes=4):
        super(HwWidarModel, self).__init__()
        self.task_type = task_type
        self.out_freq_dim = out_freq_dim
        self.time_dim = time_dim
        self.doppler_in_channels = doppler_in_channels

        # ========== 1) Doppler 分支: CNN + Adaptive Pooling + MaxPool + LSTM ==========
        self.conv1 = nn.Conv2d(doppler_in_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((out_freq_dim, time_dim))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
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
        doppler_hidden_batch = []
        for i in range(B):
            x = doppler_list[i].to(next(self.parameters()).device)  # shape=(2,freq_dim_i, 4)
            x = x.unsqueeze(0)  # =>(1,2,freq_dim_i,4)
            # CNN
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)

            x = self.adaptive_pool(x)  # =>(1,8,out_freq_dim, time_dim)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.pool(x)  # =>(1,16,out_freq_dim//2,time_dim//2)

            x = self.dropout_cnn(x)  # CNN后再 dropout

            # Reshape => LSTM
            # x.shape=(1,16,out_freq_dim//2, time_dim//2)
            _, C, Fh, Tw = x.shape
            x = x.view(1, C * Fh, Tw)  # =>(1, 16*(out_freq_dim//2), time_dim//2)
            x = x.permute(0, 2, 1)  # =>(1, time_dim//2, 16*(out_freq_dim//2))

            out_d, _ = self.doppler_lstm(x)  # =>(1, time_dim//2, doppler_hidden_size)
            hidden_doppler = out_d[:, -1, :]  # =>(1, doppler_hidden_size)
            hidden_doppler = self.dropout_lstm(hidden_doppler)
            doppler_hidden_batch.append(hidden_doppler)

        doppler_batch = torch.cat(doppler_hidden_batch, dim=0)  # =>(B, doppler_hidden_size)

        # 3) 融合
        fusion = torch.cat([doppler_batch, rssi_batch], dim=1)  # =>(B, doppler_hidden_size + rssi_hidden_size)
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


class SubModel(nn.Module):
    def __init__(self, input_shape, n_class, f_dropout_ratio=0.5, f_learning_rate=0.001):
        super(SubModel, self).__init__()
        self.f_dropout_ratio = f_dropout_ratio
        self.f_learning_rate = f_learning_rate

        # 计算Flatten层的输入维度
        t_max = input_shape[0]
        self.flatten_dim = (t_max - 4) * 2 * 58 * 16

        # 定义网络层
        self.conv3d = nn.Conv3d(1, 16, kernel_size=(5, 3, 6), padding=0)
        self.maxpool3d = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.dropout1 = nn.Dropout(f_dropout_ratio)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(f_dropout_ratio)
        self.fc3 = nn.Linear(128, n_class)

        # 定义优化器和损失函数
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=f_learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # 确保输入有正确的通道维度
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 添加通道维度 (batch, T_MAX, 6, 121) -> (batch, 1, T_MAX, 6, 121)

        # CNN部分
        x = nn.functional.relu(self.conv3d(x))
        x = self.maxpool3d(x)

        # 展平
        x = x.view(-1, self.flatten_dim)

        # 全连接层
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = nn.functional.relu(self.fc2(x))
        x = self.dropout2(x)
        x = nn.functional.softmax(self.fc3(x), dim=1)

        return x
