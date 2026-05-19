import os
import sys
import numpy as np
import yaml
import logging
from typing import Tuple, Dict, Any

from src.math.svd import compute_standard_svd, choose_energy_threshold
from src.math.condition_number import calculate_condition_number, evaluate_stability
from src.anomaly.reconstruction import reconstruct_matrix
from src.anomaly.residual_score import calculate_anomaly_score
from src.storage.mlflow_tracking import MLflowTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchPipeline")

class BatchPipelineManager:
    """
    Manages offline batch training and validation using historical datasets 
    (NASA Bearing, SECOM, or PHM). Establishes nominal baselines and static thresholds.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.k_components = self.config['models']['incremental_svd']['n_components']
        self.mlflow_tracker = MLflowTracker()
        
        # Save directory
        self.save_dir = "artifacts"
        os.makedirs(self.save_dir, exist_ok=True)

    def load_historical_dataset(self, dataset_name: str = "NASA_Bearing") -> np.ndarray:
        """
        Loads or generates historical baseline sensor metrics representing nominal behavior.
        
        Simulates the NASA Bearing dataset (high frequency vibration matrix, 4 bearings x time steps)
        or the SECOM dataset (semi-conductor manufacturing process variables).
        """
        logger.info(f"Loading historical dataset: '{dataset_name}'...")
        
        # Generate high-fidelity nominal baseline
        # In a real environment, we would load the actual raw files from NASA IMS Bearing Dataset
        n_sensors = 10  # matching input_dim in config
        n_timestamps = 1000
        
        np.random.seed(42)
        # Establish standard nominal mechanical vibration with sinusoidal motor harmonics
        t = np.linspace(0, 10, n_timestamps)
        base_vibration = np.zeros((n_sensors, n_timestamps))
        
        for i in range(n_sensors):
            # Harmonics corresponding to bearing pass frequencies (BPFO / BPFI)
            freq = 2.0 + (i * 0.5)
            base_vibration[i, :] = 0.5 * np.sin(2 * np.pi * freq * t) + np.random.normal(0.0, 0.1, n_timestamps)
            
        return base_vibration

    def train_baseline(self, dataset_name: str = "NASA_Bearing") -> Dict[str, Any]:
        """
        Runs batch SVD to compute baseline parameters and nominal statistical limits.
        """
        logger.info("Starting batch baseline training...")
        self.mlflow_tracker.start_run(run_name="Batch_Baseline_Training")
        
        # Load data
        X = self.load_historical_dataset(dataset_name)
        n_sensors, n_time = X.shape
        logger.info(f"Loaded feature matrix X of shape {X.shape} (sensors={n_sensors}, timestamps={n_time})")
        
        # 1. Compute Full SVD on the nominal baseline matrix
        U, S, Vt = compute_standard_svd(X)
        
        # 2. Check stability and condition number
        cond = calculate_condition_number(S)
        stability = evaluate_stability(cond)
        logger.info(f"Baseline Condition Number: {cond:.2f} | Status: {stability['status']}")
        
        # 3. Determine dynamic rank k based on energy threshold (95% energy)
        k_opt = choose_energy_threshold(S, target_energy=0.95)
        logger.info(f"Optimal truncation rank (95% energy threshold): k = {k_opt}")
        
        # 4. Reconstruct nominal matrix to find normal reconstruction error distribution
        X_hat = reconstruct_matrix(U, S, Vt, k=k_opt)
        
        # Calculate nominal anomaly score distribution
        scores = []
        # Calculate scores over a sliding window across the dataset
        window_size = 100
        step_size = 20
        
        for start_idx in range(0, n_time - window_size, step_size):
            X_window = X[:, start_idx : start_idx + window_size]
            X_hat_window = X_hat[:, start_idx : start_idx + window_size]
            
            res_dict = calculate_anomaly_score(X_window, X_hat_window)
            scores.append(res_dict["normalized_residual"])
            
        scores = np.array(scores)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        
        # Compute static Mean + 3 * Std threshold
        static_threshold = mean_score + 3.0 * std_score
        logger.info(f"Nominal Score Distribution: Mean = {mean_score:.4f}, Std = {std_score:.4f}")
        logger.info(f"Established Static Alarm Threshold (Mean + 3*Std) = {static_threshold:.4f}")
        
        # Save offline parameters to Yaml artifact
        baseline_params = {
            "dataset_name": dataset_name,
            "trained_at": time.time(),
            "n_sensors": int(n_sensors),
            "k_optimal": int(k_opt),
            "condition_number": float(cond),
            "nominal_score_mean": float(mean_score),
            "nominal_score_std": float(std_score),
            "static_threshold": float(static_threshold)
        }
        
        artifact_path = os.path.join(self.save_dir, "svd_baseline_parameters.yaml")
        with open(artifact_path, "w") as f:
            yaml.safe_dump(baseline_params, f)
            
        logger.info(f"Saved baseline parameters yaml to {artifact_path}")
        
        # MLflow Tracking
        self.mlflow_tracker.log_params(baseline_params)
        self.mlflow_tracker.log_metric("trained_static_threshold", static_threshold, step=0)
        self.mlflow_tracker.end_run()
        
        return baseline_params

if __name__ == "__main__":
    manager = BatchPipelineManager()
    manager.train_baseline()
