import numpy as np
from src.math.norms import compute_normalized_residual, frobenius_norm
from typing import Dict, Union

def calculate_anomaly_score(X: np.ndarray, X_hat: np.ndarray) -> Dict[str, float]:
    """
    Computes both absolute and normalized anomaly scores based on SVD reconstruction residual.
    
    score_raw = ||X - X_hat||_F
    score_normalized = ||X - X_hat||_F / ||X||_F
    
    Returns:
      A dictionary containing:
        - "raw_residual": Absolute Frobenius norm of the reconstruction error.
        - "normalized_residual": Frobenius norm normalized by the original signal norm.
    """
    raw_res = frobenius_norm(X - X_hat)
    norm_res = compute_normalized_residual(X, X_hat)
    
    return {
        "raw_residual": raw_res,
        "normalized_residual": norm_res
    }
