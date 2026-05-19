import logging
from typing import Dict, Any

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = logging.getLogger("MLflowTracking")

class MLflowTracker:
    """
    Handles MLOps experiment tracking using MLflow.
    Logs SVD singular components, thresholds, and performance scores.
    """
    def __init__(self, experiment_name: str = "SVD_Predictive_Maintenance"):
        self.experiment_name = experiment_name
        self.active_run = None
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(self.experiment_name)
                logger.info(f"MLflow experiment set to '{self.experiment_name}'")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}. MLflow logging disabled.")
                self.mock = True
        else:
            logger.warning("MLflow package not installed. Experiment tracking is simulated locally.")

    def start_run(self, run_name: str = "RealTime_Pipeline") -> None:
        """
        Starts an MLflow tracking run.
        """
        if MLFLOW_AVAILABLE:
            try:
                self.active_run = mlflow.start_run(run_name=run_name)
                logger.info(f"Started MLflow run '{run_name}'")
            except Exception as e:
                logger.error(f"Error starting MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Logs a dictionary of static configuration hyperparameters.
        """
        if MLFLOW_AVAILABLE and self.active_run:
            try:
                mlflow.log_params(params)
            except Exception as e:
                logger.error(f"MLflow param logging failed: {e}")
        else:
            logger.info(f"[MLflow Sim Log Params] {params}")

    def log_metric(self, key: str, value: float, step: int) -> None:
        """
        Logs numerical metrics (e.g. reconstruction error, condition number, latency).
        """
        if MLFLOW_AVAILABLE and self.active_run:
            try:
                mlflow.log_metric(key, float(value), step=step)
            except Exception as e:
                logger.error(f"MLflow metric logging failed: {e}")
        else:
            # We don't flood console logs for step-wise metrics unless debug is enabled,
            # but we can print them when critical warnings/alerts trigger.
            pass

    def end_run(self) -> None:
        """
        Gracefully ends the current MLflow run.
        """
        if MLFLOW_AVAILABLE and self.active_run:
            try:
                mlflow.end_run()
                logger.info("Ended MLflow tracking run.")
                self.active_run = None
            except Exception as e:
                logger.error(f"Error closing MLflow run: {e}")
