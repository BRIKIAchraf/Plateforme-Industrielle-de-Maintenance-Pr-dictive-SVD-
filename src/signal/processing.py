import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple

class IndustrialSignalProcessor:
    """
    Advanced industrial signal processing suite for filtering, detrending,
    frequency-domain transformation, and wavelet denoising.
    """
    def __init__(self, sampling_rate_hz: float = 1000.0):
        self.fs = sampling_rate_hz

    def butterworth_lowpass(self, data: np.ndarray, cutoff_hz: float, order: int = 5) -> np.ndarray:
        """
        Apply a Butterworth low-pass filter along the time axis (axis 0).
        """
        nyquist = 0.5 * self.fs
        normal_cutoff = cutoff_hz / nyquist
        # Get butterworth filter coefficients
        b, a = scipy_signal.butter(order, normal_cutoff, btype='low', analog=False)
        # Apply zero-phase filter to prevent time delay/phase shift
        return scipy_signal.filtfilt(b, a, data, axis=0)

    def detrend_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Remove linear trend from the signal to avoid low-frequency baseline drifts.
        """
        return scipy_signal.detrend(data, axis=0)

    def compute_fft(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Fast Fourier Transform (FFT) of the input data (time_steps, channels).
        Returns:
            frequencies: 1D array of frequency bins
            amplitudes: 2D array of FFT amplitude spectrum
        """
        n = len(data)
        freqs = np.fft.rfftfreq(n, d=1/self.fs)
        fft_vals = np.fft.rfft(data, axis=0)
        amplitudes = np.abs(fft_vals) / n
        return freqs, amplitudes

    def haar_wavelet_denoise(self, data: np.ndarray, level: int = 1) -> np.ndarray:
        """
        Perform high-performance 1D Haar Wavelet Denoising in pure NumPy.
        Decomposes signal into scaling/detail coefficients, thresholds the details,
        and reconstructs the denoised signal.
        """
        denoised = np.copy(data)
        for col in range(data.shape[1]):
            denoised[:, col] = self._haar_denoise_1d(data[:, col], level)
        return denoised

    def _haar_denoise_1d(self, x: np.ndarray, level: int) -> np.ndarray:
        n = len(x)
        # Pad to next power of 2 if necessary to ensure clean decomposition
        pow2 = 2**int(np.ceil(np.log2(n)))
        if pow2 > n:
            padded = np.pad(x, (0, pow2 - n), mode='edge')
        else:
            padded = np.copy(x)

        # Decompose
        coeffs = []
        arr = padded
        for _ in range(level):
            # Haar lowpass/highpass operations
            approx = (arr[0::2] + arr[1::2]) / np.sqrt(2)
            detail = (arr[0::2] - arr[1::2]) / np.sqrt(2)
            coeffs.append(detail)
            arr = approx
        
        # Simple Soft Thresholding on detail coefficients (Universal threshold estimation)
        # threshold = sigma * sqrt(2 * ln(N))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 1.0
        thresh = sigma * np.sqrt(2 * np.log(len(padded)))

        for i in range(len(coeffs)):
            detail = coeffs[i]
            coeffs[i] = np.sign(detail) * np.maximum(np.abs(detail) - thresh, 0.0)

        # Reconstruct
        arr_rec = arr
        for detail in reversed(coeffs):
            # Interleave approximation and detail
            up_approx = np.zeros(len(arr_rec) * 2)
            up_detail = np.zeros(len(arr_rec) * 2)
            up_approx[0::2] = arr_rec / np.sqrt(2)
            up_approx[1::2] = arr_rec / np.sqrt(2)
            up_detail[0::2] = detail / np.sqrt(2)
            up_detail[1::2] = -detail / np.sqrt(2)
            arr_rec = up_approx + up_detail

        # Slice back to original size
        return arr_rec[:n]

    def preprocess_pipeline(self, raw_data: np.ndarray, lowpass_cutoff: float = 150.0) -> np.ndarray:
        """
        Complete signal preprocessing pipeline:
        1. Detrend signal (removes mechanical/sensor slide drift)
        2. Butterworth low-pass filter (removes high-frequency ambient electrical noise)
        3. Haar Wavelet denoise (removes non-stationary spikes)
        """
        step1 = self.detrend_signal(raw_data)
        step2 = self.butterworth_lowpass(step1, cutoff_hz=lowpass_cutoff)
        step3 = self.haar_wavelet_denoise(step2, level=2)
        return step3
