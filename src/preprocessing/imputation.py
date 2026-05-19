import numpy as np
import pandas as pd
from typing import Union, List

class KalmanFilter1D:
    """
    A simple 1D Kalman Filter to track state and filter out high-frequency noise/spikes,
    as well as impute missing sensor measurements.
    """
    def __init__(self, q: float = 1e-4, r: float = 1e-2):
        self.q = q  # Process noise covariance
        self.r = r  # Measurement noise covariance
        self.x = None  # Estimated state
        self.p = 1.0   # Estimation error covariance

    def update(self, z: Union[float, None]) -> float:
        """
        Update the filter state with a new measurement z.
        If z is None or NaN, we perform a prediction step only (imputation).
        """
        # 1. Prediction step
        # x_pred = x (identity state transition model)
        # p_pred = p + q
        if self.x is None:
            if z is not None and not np.isnan(z):
                self.x = z
                self.p = 1.0
                return self.x
            else:
                return 0.0  # Default fallback if first point is missing

        x_pred = self.x
        p_pred = self.p + self.q

        if z is None or np.isnan(z):
            # Measurement is missing! Impute using predicted state
            self.x = x_pred
            self.p = p_pred
            return self.x

        # 2. Update step
        # Kalman Gain
        k = p_pred / (p_pred + self.r)
        # Updated state estimate
        self.x = x_pred + k * (z - x_pred)
        # Updated error covariance
        self.p = (1.0 - k) * p_pred
        
        return self.x

class MissingDataEngine:
    """
    Industrial preprocessing and data quality engine to clean, impute and smooth
    multivariate sensor readings.
    """
    def __init__(self, kalman_q: float = 1e-4, kalman_r: float = 1e-2):
        self.kalman_q = kalman_q
        self.kalman_r = kalman_r
        self._filters = {}  # One Kalman Filter per sensor channel

    def impute_basic(self, df: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """
        Perform basic imputation using standard forward-fill or linear interpolation.
        """
        if method == "ffill":
            return df.ffill().bfill()
        elif method == "interpolate":
            return df.interpolate(method="linear").ffill().bfill()
        else:
            raise ValueError(f"Unknown imputation method: {method}")

    def clean_sensor_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Cleans a 2D numpy array (time_steps, channels).
        Applies Kalman filtering to smooth sensor noise and fill missing values.
        """
        time_steps, channels = X.shape
        X_cleaned = np.zeros_like(X)

        for col in range(channels):
            if col not in self._filters:
                self._filters[col] = KalmanFilter1D(q=self.kalman_q, r=self.kalman_r)
            
            kf = self._filters[col]
            for t in range(time_steps):
                val = X[t, col]
                X_cleaned[t, col] = kf.update(val)
                
        return X_cleaned
