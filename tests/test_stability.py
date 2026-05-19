import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.math.condition_number import calculate_condition_number, evaluate_stability
from src.math.sherman_morrison import sherman_morrison_update

def test_condition_number_values():
    # Stable matrix: identity matrix has condition number = 1.0
    S_stable = np.array([1.0, 1.0, 1.0])
    cond_stable = calculate_condition_number(S_stable)
    assert cond_stable == 1.0
    
    # Singularity near zero: very large condition number
    S_singular = np.array([5.0, 1.0, 0.0])
    cond_singular = calculate_condition_number(S_singular)
    assert cond_singular >= 1e10

def test_stability_evaluation():
    res_stable = evaluate_stability(50.0)
    res_moderate = evaluate_stability(500.0)
    res_unstable = evaluate_stability(5000.0)
    
    assert res_stable["status"] == "STABLE"
    assert res_moderate["status"] == "MODERATE"
    assert res_unstable["status"] == "UNSTABLE"

def test_sherman_morrison_update():
    np.random.seed(42)
    n = 4
    
    # Create random invertible matrix A
    A = np.random.normal(0, 1, (n, n))
    A = np.dot(A, A.T) + np.eye(n) * 0.1  # Make symmetric positive-definite (invertible)
    
    A_inv = np.linalg.inv(A)
    
    # Rank-1 update vectors
    u = np.random.normal(0, 0.5, (n, 1))
    v = np.random.normal(0, 0.5, (n, 1))
    
    # Updated matrix A_new = A + u * v^T
    A_new = A + np.dot(u, v.T)
    A_new_inv_expected = np.linalg.inv(A_new)
    
    # Apply Sherman-Morrison
    A_new_inv_actual = sherman_morrison_update(A_inv, u, v)
    
    assert np.allclose(A_new_inv_expected, A_new_inv_actual, atol=1e-7)
