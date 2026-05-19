import numpy as np

def sherman_morrison_update(A_inv: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Applies the Sherman-Morrison formula to compute the inverse of a matrix after a rank-1 update.
    (A + u v^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u)
    
    This is extremely useful for streaming high-throughput sensor streams, enabling
    incremental updates to the covariance/correlation matrix inverse in O(n^2) instead of O(n^3).
    
    Parameters:
      A_inv: The inverse of matrix A, shape (n, n)
      u: Vector of shape (n, 1) or (n,) representing the rank-1 update column
      v: Vector of shape (n, 1) or (n,) representing the rank-1 update row
      
    Returns:
      A_inv_new: The updated inverse matrix, shape (n, n)
    """
    # Ensure u and v are 2D column vectors for matrix multiplication
    u = np.atleast_2d(u)
    if u.shape[0] < u.shape[1]:
        u = u.T  # Shape (n, 1)
        
    v = np.atleast_2d(v)
    if v.shape[0] < v.shape[1]:
        v = v.T  # Shape (n, 1)
        
    n = A_inv.shape[0]
    if u.shape[0] != n or v.shape[0] != n:
        raise ValueError(f"Vectors u and v must match the dimension of A_inv ({n}x{n}). Got u: {u.shape}, v: {v.shape}")
        
    # Compute the denominator: 1 + v^T * A_inv * u
    # A_inv * u has shape (n, 1)
    A_inv_u = np.dot(A_inv, u)
    
    # v^T * A_inv * u is a scalar
    denominator = 1.0 + np.dot(v.T, A_inv_u)[0, 0]
    
    if np.abs(denominator) < 1e-12:
        # Fallback or raise warning if division by zero occurred (singular update)
        raise ValueError("Sherman-Morrison update failed: Denominator (1 + v^T A_inv u) is extremely close to zero, causing singular system.")
        
    # Compute numerator: A_inv * u * v^T * A_inv
    # v^T * A_inv has shape (1, n)
    v_T_A_inv = np.dot(v.T, A_inv)
    
    # Outer product (n, 1) x (1, n) -> (n, n)
    numerator = np.dot(A_inv_u, v_T_A_inv)
    
    # Update inverse
    A_inv_new = A_inv - numerator / denominator
    return A_inv_new
