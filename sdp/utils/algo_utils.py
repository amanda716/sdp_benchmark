
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from pandas import DataFrame
import pandas as pd
from typing import List, Dict
from scipy.signal import butter, filtfilt
import torch


def db_to_power(x):
    """
    Converts a value from decibels (dB) to linear power.
    The formula used for the conversion is:
        power = 10^(x / 10)
    Parameters:
    x (float): The value in decibels (dB) to be converted.
    Returns:
    float: The equivalent linear power.
    """

    return 10**(x / 10.0)


def get_total_rss(rssi):
    """
    Calculate the total received signal strength (RSS) in decibels (dB) 
    from a list of RSSI values.
    This function converts each RSSI value from dB to linear scale, computes 
    the average of the linear values, and then converts the result back to dB.
    Args:
        rssi (list of float): A list of RSSI values in decibels (dB).
    Returns:
        float: The total RSS in decibels (dB).
    """

    linear_vals = [db_to_power(val) for val in rssi]
    avg_linear_rss = np.mean(linear_vals)
    return 10 * np.log10(avg_linear_rss)

# TODO: below function needs to be refactored later


def interpolate_csi(df: DataFrame, target_time, num_subcarriers=30, num_antennas=4):
    """
    对 CSI 数据进行插值，统一采样率到100 Hz。

    参数:
    - df: 包含 CSI 数据的 DataFrame，必须包含 'total_seconds' 和 'data' 列
    - target_time: 均匀时间网格数组
    - num_subcarriers: 子载波数（默认为30）
    - num_antennas: 天线数（默认为4）

    返回:
    - csi_uniform: 插值后的 CSI 数据，形状为 (len(target_time), num_subcarriers, num_antennas)
    """
    T = len(target_time)
    csi_uniform = np.zeros(
        (T, num_subcarriers, num_antennas), dtype=np.complex128)

    original_time = df['total_seconds'].values

    for sc in range(num_subcarriers):
        for ant in range(num_antennas):
            # 提取原始 CSI 数据
            csi_complex = np.array([rec.csi_matrix[sc, ant] for rec in df['data']])
            # 提取实部和虚部
            csi_real = csi_complex.real
            csi_imag = csi_complex.imag

            # 创建插值函数
            interp_real = interp1d(
                original_time, csi_real, kind='linear', fill_value='extrapolate')
            interp_imag = interp1d(
                original_time, csi_imag, kind='linear', fill_value='extrapolate')

            # 插值到均匀时间网格
            csi_real_uniform = interp_real(target_time)
            csi_imag_uniform = interp_imag(target_time)

            # 重构复数 CSI 数据
            csi_uniform[:, sc, ant] = csi_real_uniform + 1j * csi_imag_uniform

    return csi_uniform


def interpolate_rssi(data, target_time, num_rssi=4):
    """
    对 RSSI 数据进行插值，统一采样率到100 Hz。

    参数:
    - data: 包含 CSI 数据的列表，每个元素包含 'rssi' 键
    - target_time: 均匀时间网格数组
    - num_rssi: 每条记录的 RSSI 数量（默认为4）

    返回:
    - rssi_uniform: 插值后的 RSSI 数据，形状为 (len(target_time), num_rssi)
    """
    T = len(target_time)
    rssi_uniform = np.zeros((T, num_rssi), dtype=np.float32)

    original_time = np.array(
        [rec.ts[0]*3600 + rec.ts[1]*60 + rec.ts[2] for rec in data])

    for r in range(num_rssi):
        # 提取第 r 个 RSSI 值，如果某条记录的 RSSI 不足 r+1 个，则使用最后一个 RSSI 值填充
        rssi_values = []
        for rec in data:
            if len(rec.rssi) > r:
                rssi_values.append(rec.rssi[r])
            else:
                rssi_values.append(rec.rssi[-1])

        # 创建插值函数
        interp_func = interp1d(original_time, rssi_values,
                               kind='linear', fill_value='extrapolate')

        # 插值到均匀时间网格
        rssi_uniform[:, r] = interp_func(target_time)

    return rssi_uniform


def get_scaled_csi_single(csi, rssi, noise_db=-92, Ntx=1, Nrx=3):
    """
    缩放 CSI 数据。

    参数:
    - csi: 原始 CSI 数据（64x4 矩阵）
    - rssi: RSSI 值列表
    - noise_db: 噪声功率（dB）
    - Ntx: 发送天线数
    - Nrx: 接收天线数

    返回:
    - scaled_csi: 缩放后的 CSI 数据
    """
    csi = csi.astype(np.complex128, copy=False)
    total_rss_db = get_total_rss(rssi)
    rssi_pwr = dbinv(total_rss_db)

    csi_sq = csi * np.conjugate(csi)
    csi_pwr = np.sum(csi_sq)
    scale = rssi_pwr / (csi_pwr / 30.0)

    thermal_noise_pwr = dbinv(noise_db)
    quant_error_pwr = scale * (Nrx * Ntx)
    total_noise_pwr = thermal_noise_pwr + quant_error_pwr

    scaled_csi = csi * np.sqrt(scale / total_noise_pwr)
    if Ntx == 2:
        scaled_csi *= np.sqrt(2)
    elif Ntx == 3:
        scaled_csi *= np.sqrt(dbinv(4.5))
    return scaled_csi


def design_bandpass_filter(lowcut, highcut, fs, order=5):
    """
    设计一个带通滤波器，并返回滤波器系数。

    参数:
    - lowcut: 带通滤波器的低截止频率（Hz）
    - highcut: 带通滤波器的高截止频率（Hz）
    - fs: 采样率（Hz）
    - order: 滤波器的阶数

    返回:
    - b, a: 滤波器系数
    """
    nyq = 0.5 * fs  # 奈奎斯特频率
    # 确保 highcut 不超过奈奎斯特频率的 99%
    highcut = min(highcut, 0.99 * nyq)
    # 确保 lowcut 大于 0，并且小于 highcut
    lowcut = max(lowcut, 0.01 * nyq)
    if lowcut >= highcut:
        raise ValueError(f"低截止频率 ({lowcut} Hz) 必须小于高截止频率 ({highcut} Hz)。")

    Wn = [lowcut / nyq, highcut / nyq]

    # 检查规范化截止频率是否在 (0,1) 之间
    if not (0 < Wn[0] < Wn[1] < 1):
        raise ValueError(
            f"规范化截止频率 Wn={Wn} 不在 (0,1) 之间。请检查 lowcut 和 highcut 的设置。")

    b, a = butter(order, Wn, btype='band')
    return b, a


def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    应用带通滤波器到数据。

    参数:
    - data: 待滤波的数据（1D NumPy 数组）
    - lowcut: 带通滤波器的低截止频率（Hz）
    - highcut: 带通滤波器的高截止频率（Hz）
    - fs: 采样率（Hz）
    - order: 滤波器的阶数

    返回:
    - y: 滤波后的数据
    """
    b, a = design_bandpass_filter(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def get_doppler_spectrum_from_tensor(csi_tensor, timestamps,
                                     rx_cnt=1, rx_acnt=4,
                                     method='stft',
                                     fs=100.0,                # 固定目标采样率
                                     window_size_seconds=2.0,
                                     hop_size_seconds=1.5):
    """
    生成多普勒谱。

    参数:
    - csi_tensor: 缩放后的CSI数据，形状为 (T, 30, 4)
    - timestamps: 时间戳数组
    - rx_cnt: 接收天线数（默认1）
    - rx_acnt: 每个接收天线的子天线数（默认4）
    - method: 生成多普勒谱的方法（默认'stft'）
    - fs: 采样率（Hz）
    - window_size_seconds: 窗口大小（秒）
    - hop_size_seconds: 窗口跳跃大小（秒）

    返回:
    - doppler_spectrum: 多普勒谱，形状为 (rx_cnt, freq_bins, time_frames)
    - freq_bin: 频率轴
    - frame_timestamps: 每帧的时间戳
    """
    samp_rate = fs
    half_rate = samp_rate / 2

    # 设定带通滤波频率
    lowcut = 2.0    # 低端2Hz
    highcut = 40.0  # 高端40Hz

    # Clamp highcut
    if highcut > 0.99 * half_rate:
        print(f"[clamp] highcut={highcut}超出Nyquist={half_rate},自动缩小")
        highcut = 0.99 * half_rate
    if lowcut < 0.01:
        lowcut = 0.01
    if lowcut >= highcut:
        lowcut = 1.0
        highcut = 0.99 * half_rate

    print(
        f"[Info] 采样率 fs={samp_rate}Hz, lowcut={lowcut}Hz, highcut={highcut}Hz")

    # 设计带通滤波器
    try:
        b_band, a_band = design_bandpass_filter(
            lowcut, highcut, samp_rate, order=5)
    except ValueError as e:
        print(f"[滤波器设计错误]: {e}")
        return np.zeros((rx_cnt, 0, 0), dtype=np.float32), None, None

    doppler_spectrum_list = []
    freq_bin = None
    frame_timestamps = []

    for ii in range(rx_cnt):
        start_antenna = ii * rx_acnt
        end_antenna = start_antenna + rx_acnt
        # 选前30子载波
        csi_data = csi_tensor[:, :30,
                              start_antenna:end_antenna].reshape(-1, 30 * rx_acnt)

        # 选择最佳天线对
        csi_mean = np.mean(np.abs(csi_data), axis=0)
        csi_var = np.sqrt(np.var(np.abs(csi_data), axis=0) + 1e-9)
        csi_mean_var_ratio = csi_mean / csi_var
        try:
            csi_mean_var_ratio_2d = csi_mean_var_ratio.reshape(
                (30, rx_acnt), order='F')
        except ValueError:
            print(f"[reshape失败], skip rx={ii}")
            continue
        idx = np.argmax(np.mean(csi_mean_var_ratio_2d, axis=0))
        start_col = idx * 30
        end_col = (idx + 1) * 30
        csi_data_ref = np.tile(csi_data[:, start_col:end_col], (1, rx_acnt))

        # 幅度调整
        csi_data_adj = np.zeros_like(csi_data, dtype=np.complex128)
        csi_data_ref_adj = np.zeros_like(csi_data_ref, dtype=np.complex128)
        alpha_sum = 0
        for jj in range(30 * rx_acnt):
            amp = np.abs(csi_data[:, jj])
            amp_nonzero = amp[amp > 0]
            alpha = np.min(amp_nonzero) if len(amp_nonzero) > 0 else 1e-6
            alpha_sum += alpha
            csi_data_adj[:, jj] = np.maximum(
                amp - alpha, 0) * np.exp(1j * np.angle(csi_data[:, jj]))
        beta = 1000 * alpha_sum / (30 * rx_acnt)
        for jj in range(30 * rx_acnt):
            amp_ref = np.abs(csi_data_ref[:, jj])
            csi_data_ref_adj[:, jj] = (
                amp_ref + beta) * np.exp(1j * np.angle(csi_data_ref[:, jj]))

        # conj_mult
        conj_mult = csi_data_adj * np.conjugate(csi_data_ref_adj)
        # 去除 idx
        conj_mult = np.concatenate(
            [conj_mult[:, :start_col], conj_mult[:, end_col:]], axis=1)

        # 带通滤波
        for jj in range(conj_mult.shape[1]):
            try:
                conj_mult[:, jj] = filtfilt(b_band, a_band, conj_mult[:, jj])
            except Exception as ex:
                print(f"[Filter失败@列{jj}]: {ex}")
                conj_mult[:, jj] = 0 + 0j

        # PCA(只保留第1主成分)
        try:
            U, S, Vh = np.linalg.svd(conj_mult, full_matrices=False)
            # conj_mult_pca => shape (T,)
            conj_mult_pca = conj_mult @ Vh.conjugate().T[:, 0]
        except np.linalg.LinAlgError as ex:
            print(f"[PCA失败]: {ex}")
            continue

        # STFT
        if method.lower() == 'stft':
            window_size_samples = int(
                round(window_size_seconds * samp_rate))  # 2.0 * 100 = 200
            hop_size_samples = int(
                round(hop_size_seconds * samp_rate))     # 1.5 * 100 = 150

            print(f"nperseg: {window_size_samples}")
            # 设置 nfft 至至少 nperseg，且通常选择 2 的幂次方
            nfft = max(window_size_samples, 512)
            print(f"nfft: {nfft}")
            window = signal.windows.gaussian(
                window_size_samples, std=window_size_samples / 6)
            f, t, Zxx = signal.stft(
                conj_mult_pca,
                fs=samp_rate,
                window=window,
                nperseg=window_size_samples,
                noverlap=window_size_samples - hop_size_samples,
                nfft=nfft,  # 明确设置 nfft
                boundary=None
            )
            print(
                f"STFT result: f.shape={f.shape}, t.shape={t.shape}, Zxx.shape={Zxx.shape}")
            freq_time_prof_allfreq = Zxx
            # 选频
            freq_lpf_sel = (f <= highcut)  # 仅保留 0 到 highcut 之间的频率
            freq_time_prof = freq_time_prof_allfreq[freq_lpf_sel, :]
            freq_bin = f[freq_lpf_sel]
            # 幅度 & 每帧归一化
            freq_time_prof = np.abs(freq_time_prof)
            sum_val = np.sum(freq_time_prof, axis=0, keepdims=True) + 1e-9
            freq_time_prof = freq_time_prof / sum_val
            frame_timestamps = t
        else:
            # cwt 略示意
            freq_bin = None
            frame_timestamps = None
            freq_time_prof = np.empty((0, 0), dtype=np.float32)

        doppler_spectrum_list.append(freq_time_prof)

    if len(doppler_spectrum_list) == 0:
        return np.zeros((rx_cnt, 0, 0), dtype=np.float32), None, None
    doppler_spectrum = np.stack(
        doppler_spectrum_list, axis=0).astype(np.float32)
    # (rx_cnt, freq_dim, time_frames)
    print(f"Doppler Spectrum shape: {doppler_spectrum.shape}")

    return doppler_spectrum, freq_bin, frame_timestamps


def doppler_collate_fn(tuple_list):
    """
    tuple_list: list of tuples (doppler_tensor, rssi_tensor, label)
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
    for (dopp_t, rssi_t, lab) in tuple_list:
        doppler_list.append(dopp_t)      # (2, freq_dim_i, 4)
        rssi_list.append(rssi_t)        # (rssi_dim_i,)
        label_list.append(lab)           # int
    label_tensor = torch.tensor(label_list, dtype=torch.long)
    return doppler_list, rssi_list, label_tensor

def dbinv(x_db):
    return 10**(x_db / 10.0)
