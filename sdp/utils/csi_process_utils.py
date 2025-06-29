import numpy as np
import pywt
import time
from scipy.stats import linregress


def batch_preprocess(csi_data_list, denoise=True, normalize=True, phase_correction=True):
    """
    预处理多个CSI样本
    返回一个包含 N 个元素的 list
    """
    print(f"Starting batch preprocessing for {len(csi_data_list)} samples...")
    start_time = time.time()
    processed_data = []

    for i, csi_data in enumerate(csi_data_list):
        try:
            processed = preprocess_csi(
                csi_data,
                denoise=denoise,
                normalize=normalize,
                phase_correction=phase_correction
            )
            processed_data.append(processed)
        except Exception as e:
            print(f"Preprocess failed on sample {i}: {e}")
            # 可以选择跳过或加入空
            processed_data.append(None)
        if (i + 1) % 10 == 0 or (i + 1) == len(csi_data_list):
            print(f"Processed {i + 1}/{len(csi_data_list)} samples")

    end_time = time.time()
    print(f"Batch preprocessing completed in {end_time - start_time:.2f} seconds")
    return processed_data

def preprocess_csi(csi_data, denoise=True, normalize=True, phase_correction=True):
    """
    预处理单条CSI数据
    结构不变，返回 [time, subcarrier, rx, tx, 2] (2=实部+虚部)
    """
    csi_data = np.array(csi_data, dtype=np.complex64, copy=True)

    if phase_correction:
        time_idx = np.arange(csi_data.shape[0])
        for sc in range(csi_data.shape[1]):
            for rx in range(csi_data.shape[2]):
                for tx in range(csi_data.shape[3]):
                    phase = np.unwrap(np.angle(csi_data[:, sc, rx, tx]), discont=np.pi)
                    if len(phase) < 5 or np.std(phase) < 1e-3:
                        continue  # 跳过非常平坦的
                    try:
                        slope, intercept, *_ = linregress(time_idx, phase)
                        if not np.isfinite(slope):
                            continue
                        csi_data[:, sc, rx, tx] *= np.exp(-1j * slope * time_idx)
                    except Exception as e:
                        print(f"Phase correction failed: sc={sc} rx={rx} tx={tx} {e}")
                        continue

    amplitude = np.abs(csi_data)
    phase = np.angle(csi_data)

    if denoise:
        def wavelet_denoise(channel):
            try:
                L = len(channel)
                if L < 8:
                    return channel
                max_level = pywt.dwt_max_level(L, pywt.Wavelet('db4').dec_len)
                level = min(2, max_level)
                if level < 1:
                    level = 1
                if L < 16:
                    level = 1
                coeffs = pywt.wavedec(channel, 'db4', level=level)
                # threshold for calculation security
                sigma = np.median(np.abs(coeffs[-level])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(L))
                denoised_coeffs = [coeffs[0]] + [
                    pywt.threshold(c, threshold, 'soft') for c in coeffs[1:]
                ]
                denoised = pywt.waverec(denoised_coeffs, 'db4')
                return denoised[:L]
            except Exception as e:
                print(f"Denoising failed: {e}")
                return channel

        for rx in range(amplitude.shape[2]):
            for tx in range(amplitude.shape[3]):
                for sc in range(amplitude.shape[1]):
                    amp_seq = amplitude[:, sc, rx, tx]
                    amplitude[:, sc, rx, tx] = wavelet_denoise(amp_seq)

    if normalize:
        reshaped = amplitude.transpose(1, 0, 2, 3).reshape(amplitude.shape[1], -1)
        amp_min = np.min(reshaped, axis=1, keepdims=True)
        amp_max = np.max(reshaped, axis=1, keepdims=True)
        amp_min = amp_min.reshape(1, -1, 1, 1)
        amp_max = amp_max.reshape(1, -1, 1, 1)
        denominator = amp_max - amp_min
        small_mask = denominator < 1e-6
        denominator[small_mask] = 1.0  # avoid divide zero
        amplitude = (amplitude - amp_min) / (denominator + 1e-10)

    processed_csi = amplitude * np.exp(1j * phase)
    return np.stack([np.real(processed_csi), np.imag(processed_csi)], axis=-1)

