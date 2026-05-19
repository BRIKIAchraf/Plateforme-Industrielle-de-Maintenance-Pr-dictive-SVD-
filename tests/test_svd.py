import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.math.svd import compute_standard_svd, compute_randomized_svd, choose_energy_threshold
from src.anomaly.reconstruction import reconstruct_matrix

def test_standard_svd_dimensions():
    # Construct a random matrix
    np.random.seed(42)
    X = np.random.normal(0, 1, (10, 20))
    
    U, S, Vt = compute_standard_svd(X)
    
    assert U.shape == (10, 10)
    assert S.shape == (10,)
    assert Vt.shape == (10, 20)
    assert np.all(S >= 0)  # Singular values must be non-negative

def test_randomized_svd_rank():
    np.random.seed(42)
    X = np.random.normal(0, 1, (50, 30))
    k = 5
    
    U, S, Vt = compute_randomized_svd(X, n_components=k)
    
    assert U.shape == (50, k)
    assert S.shape == (k,)
    assert Vt.shape == (k, 30)

def test_energy_threshold_selection():
    # Singular values where first component dominates heavily (96% of variance)
    S = np.array([10.0, 1.0, 0.5, 0.1])
    
    k_95 = choose_energy_threshold(S, target_energy=0.95)
    k_99 = choose_energy_threshold(S, target_energy=0.99)
    
    assert k_95 == 1  # 100 / 101.26 = 98.7% which is >= 95%
    assert k_99 == 2  # 101 / 101.26 = 99.7% which is >= 99%

def test_reconstruction_fidelity():
    np.random.seed(42)
    X = np.random.normal(0, 1, (8, 8))
    
    # Compute SVD
    U, S, Vt = compute_standard_svd(X)
    
    # Reconstruct at full rank (k=8)
    X_hat = reconstruct_matrix(U, S, Vt, k=8)
    
    # Check that difference is nearly zero
    assert np.allclose(X, X_hat, atol=1e-7)
