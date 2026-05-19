import numpy as np

def frobenius_norm(X: np.ndarray) -> float:
    """
    Computes the Frobenius norm of matrix X.
    ||X||_F = sqrt(sum(|x_ij|^2))
    """
    return float(np.linalg.norm(X, ord='fro'))

def l2_norm(x: np.ndarray) -> float:
    """
    Computes the L2 Euclidean norm of vector x.
    """
    return float(np.linalg.norm(x, ord=2))

def compute_normalized_residual(X: np.ndarray, X_hat: np.ndarray) -> float:
    """
    Computes the normalized reconstruction error.
    score = ||X - X_hat||_F / ||X||_F
    
    This normalization is crucial in industrial environments to handle sensor gain
    variations, noise offsets, and prevent amplitude scaling from throwing off anomaly thresholds.
    """
    norm_X = frobenius_norm(X)
    if norm_X < 1e-9:
        # Fallback to absolute Frobenius norm if matrix is nearly zero
        return frobenius_norm(X - X_hat)
        
    return frobenius_norm(X - X_hat) / norm_X
