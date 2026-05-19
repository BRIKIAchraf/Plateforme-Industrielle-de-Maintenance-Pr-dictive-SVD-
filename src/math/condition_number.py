import numpy as np
from typing import Dict, Union

def calculate_condition_number(singular_values: np.ndarray) -> float:
    """
    Computes the condition number of a matrix using its singular values.
    cond = sigma_max / sigma_min
    
    If sigma_min is close to zero, returns a large representative number (1e10)
    to reflect extreme singularity or numerical instability.
    """
    if singular_values is None or len(singular_values) == 0:
        return 1.0
        
    sigma_max = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    
    if sigma_min < 1e-12:
        return 1e10
        
    return sigma_max / sigma_min

def evaluate_stability(condition_number: float) -> Dict[str, Union[str, float]]:
    """
    Classifies the numerical stability of the linear system based on its condition number.
    
    Thresholds:
      - cond < 100: Stable / Green (OK)
      - 100 <= cond <= 1000: Moderate / Yellow (Warning)
      - cond > 1000: Unstable / Red (Numerical issues / Singularity risk)
    """
    if condition_number < 100.0:
        status = "STABLE"
        action = "None"
    elif condition_number <= 1000.0:
        status = "MODERATE"
        action = "Monitor sensor variance or inject small regularization (ridge/shrinkage)."
    else:
        status = "UNSTABLE"
        action = "High risk of numerical instability. Check for duplicate sensors, zero-signal channels, or complete failures."
        
    return {
        "condition_number": condition_number,
        "status": status,
        "action": action
    }
