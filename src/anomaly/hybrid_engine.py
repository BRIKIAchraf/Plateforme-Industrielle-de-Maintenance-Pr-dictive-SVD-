import numpy as np
from scipy.spatial.distance import mahalanobis
from typing import Dict, Tuple

class HybridAnomalyEngine:
    """
    State-of-the-art Multi-score Anomaly Engine fusing multiple physical
    and mathematical signals to robustly flag machinery failure modes.
    """
    def __init__(self, ewma_alpha: float = 0.1, initial_threshold: float = 3.0):
        self.alpha = ewma_alpha
        self.k = initial_threshold
        
        # State tracking for EWMA thresholding
        self.running_mean = None
        self.running_std = None
        
        # Covariance state for Mahalanobis distance calculations
        self.mean_vector = None
        self.inv_covariance = None

    def calculate_frequency_entropy(self, amplitudes: np.ndarray) -> float:
        """
        Compute Spectral Entropy to capture vibration irregularity.
        amplitudes: (frequencies, channels)
        """
        # Normalize FFT amplitudes to form a probability distribution
        norm_amps = amplitudes / (np.sum(amplitudes, axis=0) + 1e-9)
        # Shannon Entropy along the frequency axis, mean across channels
        entropy_per_channel = -np.sum(norm_amps * np.log2(norm_amps + 1e-9), axis=0)
        return float(np.mean(entropy_per_channel))

    def update_mahalanobis_covariance(self, X: np.ndarray) -> None:
        """
        Update the correlation matrix to calculate multivariate Mahalanobis distance.
        """
        self.mean_vector = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + np.eye(X.shape[1]) * 1e-5  # Add small ridge to avoid singularity
        self.inv_covariance = np.linalg.pinv(cov)

    def calculate_mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Calculate Mahalanobis distance for a single multi-sensor observation vector x.
        """
        if self.inv_covariance is None or self.mean_vector is None:
            return 1.0  # Fallback if covariance state is not yet warmed up
        try:
            return mahalanobis(x, self.mean_vector, self.inv_covariance)
        except Exception:
            return 1.0

    def compute_trend_slope(self, recent_scores: np.ndarray) -> float:
        """
        Measure the rate of progressive degradation (progressive wear)
        by calculating the linear regression slope of recent anomaly scores.
        """
        if len(recent_scores) < 3:
            return 0.0
        x = np.arange(len(recent_scores))
        slope, _ = np.polyfit(x, recent_scores, 1)
        return float(slope)

    def fuse_anomaly_scores(
        self, 
        svd_residual: float, 
        ae_residual: float, 
        mahalanobis_dist: float, 
        freq_entropy: float,
        spectral_drift: float
    ) -> float:
        """
        Combines different indicators into a unified multi-score Health Index.
        """
        # Weighted combination of physical indicators
        # Weights are structured based on normal relative magnitudes
        health_index = (
            0.25 * svd_residual + 
            0.25 * ae_residual + 
            0.20 * mahalanobis_dist + 
            0.15 * freq_entropy + 
            0.15 * spectral_drift
        )
        return float(health_index)

    def update_ewma_threshold(self, current_score: float) -> Tuple[float, bool]:
        """
        Applies dynamic Exponentially Weighted Moving Average (EWMA) thresholding.
        Returns:
            threshold: Current adaptive alert threshold
            is_anomaly: True if the current score exceeds the threshold
        """
        if self.running_mean is None:
            self.running_mean = current_score
            self.running_std = 1.0  # Set default initial variance
            return self.running_mean + self.k * self.running_std, False

        # Update running mean and running variance with Exponential Weighting
        self.running_mean = self.alpha * current_score + (1.0 - self.alpha) * self.running_mean
        
        diff = current_score - self.running_mean
        running_var = self.alpha * (diff ** 2) + (1.0 - self.alpha) * (self.running_std ** 2)
        self.running_std = np.sqrt(np.maximum(running_var, 1e-5))

        # Dynamic Threshold calculation
        threshold = self.running_mean + self.k * self.running_std
        is_anomaly = current_score > threshold

        return float(threshold), bool(is_anomaly)
