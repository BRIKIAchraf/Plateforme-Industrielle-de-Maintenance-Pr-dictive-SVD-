import numpy as np
from typing import Tuple, Optional

class StaticThresholdEvaluator:
    """
    Computes statistical threshold based on offline baseline history:
    threshold = mean + 3 * std
    """
    def __init__(self, mean: float, std: float, k: float = 3.0):
        self.mean = mean
        self.std = std
        self.k = k
        self.threshold = mean + k * std

    def evaluate(self, score: float) -> bool:
        """
        Returns True if the score exceeds the static threshold.
        """
        return score > self.threshold


class AdaptiveEWMAThresholding:
    """
    State-of-the-art dynamic thresholding using Exponentially Weighted Moving Average (EWMA).
    Adapts the alert boundary in real time to capture machinery drift and environmental changes.
    """
    def __init__(self, alpha: float = 0.1, k: float = 3.0):
        self.alpha = alpha
        self.k = k
        
        self.running_mean: Optional[float] = None
        self.running_std: Optional[float] = None

    def update(self, score: float) -> Tuple[float, bool]:
        """
        Updates running statistical metrics with the current score and returns:
          threshold: The computed dynamic threshold (mean + k * std)
          is_anomaly: True if the score exceeds the threshold
        """
        if self.running_mean is None:
            self.running_mean = score
            self.running_std = 1e-3  # Non-zero default variance
            return self.running_mean + self.k * self.running_std, False
            
        # Calculate dynamic threshold based on prior step's statistics
        threshold = self.running_mean + self.k * self.running_std
        is_anomaly = score > threshold
        
        # EWMA update of mean
        self.running_mean = self.alpha * score + (1.0 - self.alpha) * self.running_mean
        
        # EWMA update of variance
        diff = score - self.running_mean
        running_var = self.alpha * (diff ** 2) + (1.0 - self.alpha) * (self.running_std ** 2)
        self.running_std = np.sqrt(max(running_var, 1e-6))
        
        return float(threshold), bool(is_anomaly)

