
import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
from pandas import DataFrame
import pandas as pd
from typing import List, Dict
from scipy.signal import butter, filtfilt


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

#TODO: below function needs to be refactored later
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
            csi_complex = np.array([rec['csi'][sc, ant] for rec in df['data']])
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
        [rec['ts'][0]*3600 + rec['ts'][1]*60 + rec['ts'][2] for rec in data])

    for r in range(num_rssi):
        # 提取第 r 个 RSSI 值，如果某条记录的 RSSI 不足 r+1 个，则使用最后一个 RSSI 值填充
        rssi_values = []
        for rec in data:
            if len(rec['rssi']) > r:
                rssi_values.append(rec['rssi'][r])
            else:
                rssi_values.append(rec['rssi'][-1])

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
