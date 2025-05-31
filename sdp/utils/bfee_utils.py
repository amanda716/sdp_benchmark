import numpy as np
from scipy.signal import filtfilt, butter


def interpolate_csi(recs, bf_times, uniform_time, Nrx=3, Ntx=1):
    """
    recs => list of dict{'csi':(30,Nrx,Ntx), ...}
    shape => (N,30,Nrx*Ntx) => 逐个子载波天线做插值
    """
    from scipy.interpolate import interp1d
    N = len(recs)
    # 先把 csi 堆叠 => shape=(N,30*Nrx*Ntx)
    csi_stack = []
    for r in recs:
        # =>(30, Nrx*Ntx)
        arr = r['csi'].reshape(30, Nrx * Ntx)
        csi_stack.append(arr)
    csi_stack = np.array(csi_stack, dtype=np.complex128)  # shape=(N,30,Nrx*Ntx)

    T = len(uniform_time)
    out_arr = np.zeros((T, 30, Nrx * Ntx), dtype=np.complex128)

    for sc in range(30):
        for ant in range(Nrx * Ntx):
            # 取原时刻的 csi
            csi_complex = csi_stack[:, sc, ant]  # shape=(N,)
            reals = csi_complex.real
            imags = csi_complex.imag
            f_real = interp1d(bf_times, reals, kind='linear', fill_value='extrapolate')
            f_imag = interp1d(bf_times, imags, kind='linear', fill_value='extrapolate')
            re_intp = f_real(uniform_time)
            im_intp = f_imag(uniform_time)
            out_arr[:, sc, ant] = re_intp + 1j * im_intp
    return out_arr


def interpolate_rssi(recs, bf_times, uniform_time):
    """
    recs => each has 'rssi_a','rssi_b','rssi_c' -> shape=(N,3)
    """
    from scipy.interpolate import interp1d
    N = len(recs)
    rssi_mat = []
    for r in recs:
        # 可能有空 => 用 max(0, x)
        a = max(0, r['rssi_a'])
        b = max(0, r['rssi_b'])
        c = max(0, r['rssi_c'])
        rssi_mat.append([a, b, c])
    rssi_mat = np.array(rssi_mat, dtype=np.float32)  # shape=(N,3)

    T = len(uniform_time)
    out_rssi = np.zeros((T, 3), dtype=np.float32)
    for ch in range(3):
        vals = rssi_mat[:, ch]
        f_rssi = interp1d(bf_times, vals, kind='linear', fill_value='extrapolate')
        out_rssi[:, ch] = f_rssi(uniform_time)
    return out_rssi


def scale_csi_block(csi_30xN, rssi3, noise_db=-92, Ntx=1, Nrx=3):
    """
    csi_30xN => (30, Nrx*Ntx)
    rssi3 => [a,b,c]
    """
    csi = csi_30xN.astype(np.complex128)
    total_rss_db = get_total_rss(rssi3)
    rssi_pwr = 10 ** (total_rss_db / 10.0)

    pwr = np.sum(csi * np.conjugate(csi))
    scale = rssi_pwr / (pwr / 30.)

    thermal = 10 ** (noise_db / 10.)
    quant = scale * (Nrx * Ntx)
    total_noise = thermal + quant

    scaled = csi * np.sqrt(scale / total_noise)
    if Ntx == 2:
        scaled *= np.sqrt(2)
    elif Ntx == 3:
        scaled *= np.sqrt(10 ** (4.5 / 10.))
    return scaled


def get_total_rss(rssi3):
    """
    rssi3 => [rssi_a, rssi_b, rssi_c], 可能有0 => 干脆替换成1
    """
    arr = []
    for v in rssi3:
        if v < 1:
            v = 1
        arr.append(10 ** (v / 10.))
    lin_avg = np.mean(arr)
    return 10 * np.log10(lin_avg)


def compute_doppler_spectrum(csi_tensor, rssi_arr, fs=1000.0):
    """
    csi_tensor: shape=(T,30,Nrx*Ntx)
    rssi_arr  : shape=(T,3)
    返回 (2, freq_bin, time_frames) => 幅度+相位
    """
    # 选择最优天线 => conj
    T = csi_tensor.shape[0]
    rx_acnt = csi_tensor.shape[2]
    data_2d = csi_tensor.reshape(T, 30 * rx_acnt)  # =>(T, 30*Nrx*Ntx)

    print(f"  [Doppler] compute => T={T}, rx_acnt={rx_acnt}")

    # 同之前 PCA+滤波
    # 1) 选天线
    mean_ = np.mean(np.abs(data_2d), axis=0)
    var_ = np.sqrt(np.var(np.abs(data_2d), axis=0) + 1e-9)
    ratio = mean_ / var_
    ratio2d = ratio.reshape(30, rx_acnt, order='F')
    idx = np.argmax(np.mean(ratio2d, axis=0))
    start_col = idx * 30
    end_col = (idx + 1) * 30
    data_ref = np.tile(data_2d[:, start_col:end_col], (1, rx_acnt))

    data_adj = np.zeros_like(data_2d, dtype=np.complex128)
    data_ref_adj = np.zeros_like(data_ref, dtype=np.complex128)

    alpha_sum = 0
    for j in range(30 * rx_acnt):
        amp = np.abs(data_2d[:, j])
        nz = amp[amp > 0]
        alpha = np.min(nz) if len(nz) > 0 else 1e-6
        alpha_sum += alpha
        data_adj[:, j] = np.maximum(amp - alpha, 0) * np.exp(1j * np.angle(data_2d[:, j]))
    beta = 1000 * alpha_sum / (30 * rx_acnt)
    for j in range(30 * rx_acnt):
        amp_r = np.abs(data_ref[:, j])
        data_ref_adj[:, j] = (amp_r + beta) * np.exp(1j * np.angle(data_ref[:, j]))

    conj_mult = data_adj * np.conjugate(data_ref_adj)
    conj_mult = np.concatenate([conj_mult[:, :start_col], conj_mult[:, end_col:]], axis=1)

    # 2) 带通滤波
    conj_filt = np.zeros_like(conj_mult, dtype=np.complex128)
    for j in range(conj_mult.shape[1]):
        conj_filt[:, j] = bandpass_2_40(conj_mult[:, j], fs)

    # 3) PCA
    try:
        U, S, Vh = np.linalg.svd(conj_filt, full_matrices=False)
        pca_1 = conj_filt @ Vh.conjugate().T[:, 0]
    except np.linalg.LinAlgError:
        pca_1 = np.zeros(T, dtype=np.complex128)

    # 4) STFT
    window_size = int(round(2.0 * fs))  # 2s
    hop_size = int(round(1.5 * fs))  # 1.5s
    from scipy.signal import stft
    f, t, Zxx = stft(pca_1, fs=fs, nperseg=window_size,
                     noverlap=(window_size - hop_size),
                     boundary=None)
    freq_sel = (f <= 40.)
    f_sel = f[freq_sel]
    Z_sel = Zxx[freq_sel, :]

    # 幅度+相位 => 幅度归一化
    amp = np.abs(Z_sel)
    sum_ = np.sum(amp, axis=0, keepdims=True) + 1e-9
    amp = amp / sum_
    phase = np.angle(Z_sel)

    spec_2d = np.stack([amp, phase], axis=0)  # =>(2, freq_bins, time_frames)
    print(f"  [Doppler] STFT => Z_sel.shape={Z_sel.shape}, spec_2d.shape={spec_2d.shape}")
    return spec_2d, f_sel, t


def bandpass_2_40(sig, fs=1000.0):
    """
    对1D信号带通滤波 [2,40] Hz
    """
    b, a = design_bandpass_filter(2.0, 40.0, fs, order=5)
    return filtfilt(b, a, sig)


def design_bandpass_filter(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs
    high_cut = min(high_cut, 0.99 * nyq)
    low_cut = max(low_cut, 0.01 * nyq)
    if low_cut >= high_cut:
        low_cut = 1.0
        high_cut = 0.99 * nyq
    Wn = [low_cut / nyq, high_cut / nyq]
    b, a = butter(order, Wn, btype='band', output='ba')
    return b, a


def build_uniform_time(tstart, tend, final_fs):
    """
    构造 [tstart, tend) 步长=1/final_fs
    """
    dt = 1.0 / final_fs
    return np.arange(tstart, tend, dt, dtype=np.float64)

def get_time_seconds(recs, unit='us'):
    """
    将 BFEE记录 list => times in second
    """
    if not recs: return np.array([])
    start_ts = recs[0]['timestamp_low']
    times=[]
    for r in recs:
        dt = (r['timestamp_low'] - start_ts)
        if unit=='us':
            dt_s= dt/1e6
        else:
            dt_s= dt/1e9
        times.append(dt_s)
    return np.array(times, dtype=np.float64)
