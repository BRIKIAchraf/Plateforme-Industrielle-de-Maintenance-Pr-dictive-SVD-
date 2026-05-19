import numpy as np
from sklearn.utils.extmath import randomized_svd
from typing import Tuple, List

def compute_standard_svd(X: np.ndarray, full_matrices: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes standard Singular Value Decomposition of matrix X.
    X: shape (m, n)
    Returns:
        U: left singular vectors, shape (m, min(m, n))
        S: singular values, shape (min(m, n),)
        Vt: right singular vectors, shape (min(m, n), n)
    """
    return np.linalg.svd(X, full_matrices=full_matrices)

def compute_randomized_svd(
    X: np.ndarray, 
    n_components: int, 
    n_iter: int = 5, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes Randomized Singular Value Decomposition for high-performance low-rank updates.
    Scales at O(m * n * log(k)) instead of O(min(m,n)^2 * max(m,n)).
    
    X: shape (m, n)
    n_components: rank target (k)
    """
    # Clamping n_components to min(X.shape)
    k = min(n_components, X.shape[0], X.shape[1])
    if k <= 0:
        raise ValueError("n_components must be a positive integer.")
    
    U, S, Vt = randomized_svd(
        X, 
        n_components=k, 
        n_iter=n_iter, 
        random_state=random_state
    )
    return U, S, Vt

def choose_energy_threshold(S: np.ndarray, target_energy: float = 0.95) -> int:
    """
    Selects the truncation rank k such that the cumulative variance (energy)
    retained is at least the target_energy (default 95%).
    
    Energy is proportional to the sum of squares of singular values (Frobenius energy).
    """
    total_energy = np.sum(S ** 2)
    if total_energy == 0:
        return 1
        
    cumulative_energy = np.cumsum(S ** 2)
    ratio = cumulative_energy / total_energy
    
    # Find first index where cumulative energy meets target_energy
    indices = np.where(ratio >= target_energy)[0]
    if len(indices) == 0:
        return len(S)
    return int(indices[0]) + 1
