import numpy as np
from typing import List, Dict, Optional

class PHMValidationEngine:
    """
    Prognostics and Health Management (PHM) evaluation engine.
    Calculates RUL, False Alarm Rates (FAR), Missed Detection Rates (MDR),
    and Early Detection Delay using ground truth failure timestamps.
    """
    def __init__(self, failure_threshold: float = 10.0):
        self.failure_threshold = failure_threshold

    def estimate_remaining_useful_life(self, score_history: List[float], time_step_seconds: float = 5.0) -> float:
        """
        Estimates the Remaining Useful Life (RUL) in seconds by fitting a linear
        degradation curve to recent anomaly scores and projecting when it will
        intersect the failure_threshold.
        """
        if len(score_history) < 5:
            return float('inf')  # Not enough degradation history to extrapolate

        # Use the last 20 steps to predict short-term degradation trajectory
        recent_scores = np.array(score_history[-20:])
        x = np.arange(len(recent_scores)) * time_step_seconds
        
        slope, intercept = np.polyfit(x, recent_scores, 1)
        
        if slope <= 0:
            return float('inf')  # No upward trend detected; machine is stable
        
        # Calculate time steps remaining until failure threshold is crossed
        current_score = recent_scores[-1]
        distance_to_fail = self.failure_threshold - current_score
        
        if distance_to_fail <= 0:
            return 0.0  # Failure has already occurred/imminent
            
        rul_seconds = distance_to_fail / slope
        return float(rul_seconds)

    def calculate_alarm_performance(
        self, 
        alerts: np.ndarray, 
        ground_truth_failures: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate key industrial alarm metrics:
        - False Alarm Rate (FAR)
        - Missed Detection Rate (MDR)
        alerts: boolean 1D array indicating whether an alarm was triggered
        ground_truth_failures: boolean 1D array indicating actual physical failure state
        """
        # True Positive, False Positive, False Negative, True Negative
        tp = np.sum((alerts == True) & (ground_truth_failures == True))
        fp = np.sum((alerts == True) & (ground_truth_failures == False))
        fn = np.sum((alerts == False) & (ground_truth_failures == True))
        tn = np.sum((alerts == False) & (ground_truth_failures == False))

        false_alarm_rate = fp / (fp + tn + 1e-9)
        missed_detection_rate = fn / (fn + tp + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)

        return {
            "false_alarm_rate": float(false_alarm_rate),
            "missed_detection_rate": float(missed_detection_rate),
            "precision": float(precision),
            "recall": float(recall)
        }

    def compute_early_detection_delay(
        self, 
        alerts: np.ndarray, 
        failure_index: int, 
        time_step_seconds: float = 5.0
    ) -> Optional[float]:
        """
        Calculates how early the system triggered its first stable alert
        prior to actual physical machine failure.
        """
        # Find all alert indices prior to the failure index
        pre_failure_alerts = np.where(alerts[:failure_index] == True)[0]
        
        if len(pre_failure_alerts) == 0:
            return None  # Failed to detect early
        
        # Locate the first alert index that sustained at least 2 consecutive alerts
        first_stable_alert = None
        for idx in pre_failure_alerts:
            if idx < len(alerts) - 1 and alerts[idx + 1] == True:
                first_stable_alert = idx
                break
                
        if first_stable_alert is None:
            first_stable_alert = pre_failure_alerts[0]

        # Calculate time delay in seconds
        delay_seconds = (failure_index - first_stable_alert) * time_step_seconds
        return float(delay_seconds)
