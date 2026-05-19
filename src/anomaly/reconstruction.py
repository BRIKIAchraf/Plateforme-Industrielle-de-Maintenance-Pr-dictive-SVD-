import numpy as np
from typing import Tuple

def reconstruct_matrix(
    U: np.ndarray, 
    S: np.ndarray, 
    Vt: np.ndarray, 
    k: int
) -> np.ndarray:
    """
    Reconstructs the matrix X from its truncated SVD elements at rank k.
    X_hat = U[:, :k] * diag(S[:k]) * Vt[:k, :]
    
    Parameters:
      U: Left singular vectors, shape (m, r)
      S: Singular values, shape (r,)
      Vt: Right singular vectors, shape (r, n)
      k: Truncation rank (k <= r)
    """
    # Ensure k is within boundaries
    r = len(S)
    k_clamped = max(1, min(k, r, U.shape[1], Vt.shape[0]))
    
    U_truncated = U[:, :k_clamped]
    S_truncated = S[:k_clamped]
    Vt_truncated = Vt[:k_clamped, :]
    
    # Reconstruct: U_k @ diag(S_k) @ Vt_k
    # We do U_truncated * S_truncated to avoid creating a large diag matrix, which is more memory efficient
    X_hat = (U_truncated * S_truncated) @ Vt_truncated
    return X_hat
