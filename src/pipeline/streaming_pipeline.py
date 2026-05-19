import time
import sys
import os
import yaml
import numpy as np
import logging

from src.ingestion.kafka_consumer import KafkaSensorConsumer
from src.ingestion.stream_buffer import RealTimeStreamBuffer
from src.math.svd import compute_randomized_svd, choose_energy_threshold
from src.math.condition_number import calculate_condition_number, evaluate_stability
from src.anomaly.reconstruction import reconstruct_matrix
from src.anomaly.residual_score import calculate_anomaly_score
from src.anomaly.thresholding import AdaptiveEWMAThresholding
from src.storage.redis_state import RedisStateManager
from src.storage.mlflow_tracking import MLflowTracker
from src.visualization.dashboard_metrics import DashboardMetricsExporter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StreamingPipeline")

class StreamingPipelineManager:
    """
    Main Real-time Streaming Pipeline coordinating ingestion, SVD computing, 
    numerical stability tracking, anomaly detection, Redis caching, MLflow tracking, 
    and dashboard visualization.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load configs
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.k_components = self.config['models']['incremental_svd']['n_components']
        self.ewma_alpha = self.config['models']['hybrid_anomaly']['ewma_alpha']
        self.initial_k_thresh = self.config['models']['hybrid_anomaly']['initial_threshold']
        
        # Initialize modules
        self.consumer = KafkaSensorConsumer(
            bootstrap_servers=self.config['kafka']['bootstrap_servers'],
            topic=self.config['kafka']['topic'],
            group_id=self.config['kafka']['group_id']
        )
        
        # Buffer window: e.g. 5 seconds sliding window
        self.buffer = RealTimeStreamBuffer(window_size_seconds=5.0, max_time_steps=80)
        
        self.anomaly_detector = AdaptiveEWMAThresholding(
            alpha=self.ewma_alpha,
            k=self.initial_k_thresh
        )
        
        self.redis_manager = RedisStateManager(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            password=self.config['redis']['password']
        )
        
        self.mlflow_tracker = MLflowTracker()
        self.dashboard_exporter = DashboardMetricsExporter(port=self.config['prometheus']['port'])
        
        self.is_running = False

    def run(self) -> None:
        """
        Executes the main pipeline execution loop.
        """
        logger.info("Initializing Streaming Pipeline components...")
        self.is_running = True
        
        # Start MLflow run
        self.mlflow_tracker.start_run(run_name="Streaming_SVD_Pipeline")
        self.mlflow_tracker.log_params({
            "k_components_target": self.k_components,
            "ewma_alpha": self.ewma_alpha,
            "initial_k_threshold": self.initial_k_thresh
        })
        
        step = 0
        try:
            logger.info("Pipeline initialized. Ingesting IoT sensor metrics...")
            while self.is_running:
                t_start = time.perf_counter()
                
                # A. Poll messages and append to the stream buffer
                raw_messages = list(self.consumer.poll_messages(timeout_seconds=0.1))
                event_count = len(raw_messages)
                
                for msg in raw_messages:
                    self.buffer.append(
                        sensor_id=msg["sensor_id"],
                        timestamp=msg["timestamp"],
                        vibration=msg["vibration"],
                        temperature=msg["temperature"]
                    )
                
                # B. Extract sliding window Feature Matrix X of shape (n_sensors, n_time_steps)
                X, sensor_ids = self.buffer.get_feature_matrix()
                
                if X.shape[0] < 2 or X.shape[1] < 2:
                    # Warmup: waiting for enough data to fill the sliding window matrix
                    time.sleep(0.5)
                    continue
                
                # C. Compute truncated Randomized SVD on Feature Matrix X
                # X: shape (n_sensors, n_time_steps)
                U, S, Vt = compute_randomized_svd(X, n_components=self.k_components)
                
                # D. Track Singular Value Spectrum and evaluate Condition Number stability
                cond = calculate_condition_number(S)
                stability = evaluate_stability(cond)
                
                # Warn if numerically unstable
                if stability["status"] == "UNSTABLE":
                    logger.warning(f"NUMERICAL MATRIX INSTABILITY DETECTED! Condition number = {cond:.2f}. Action required: {stability['action']}")
                
                # E. Select rank k using dynamic energy threshold (default 95% energy)
                k_opt = choose_energy_threshold(S, target_energy=0.95)
                
                # F. Low-rank Matrix Reconstruction
                X_hat = reconstruct_matrix(U, S, Vt, k=k_opt)
                
                # G. Calculate anomaly score (raw and normalized Frobenius residuals)
                scores = calculate_anomaly_score(X, X_hat)
                score_norm = scores["normalized_residual"]
                
                # H. Adaptive EWMA Alerting Engine
                threshold, is_alarm = self.anomaly_detector.update(score_norm)
                
                # Estimate Remaining Useful Life (RUL)
                # Stable is represented as inf (9999.0 in Redis/Prometheus), declining if alarm triggered
                rul = 3600.0 / (1.0 + (step * 0.05)) if is_alarm else float('inf')
                
                t_end = time.perf_counter()
                latency_ms = (t_end - t_start) * 1000.0
                throughput = event_count / max(1e-9, (t_end - t_start))
                
                # I. Storage updates (Redis Real-time status & MLflow MLOps telemetry)
                self.redis_manager.save_metrics(
                    step=step,
                    anomaly_score=score_norm,
                    threshold=threshold,
                    is_alarm=is_alarm,
                    condition_number=cond,
                    singular_values=S,
                    rul=rul
                )
                
                self.mlflow_tracker.log_metric("anomaly_score", score_norm, step=step)
                self.mlflow_tracker.log_metric("condition_number", cond, step=step)
                self.mlflow_tracker.log_metric("reconstruction_rank", k_opt, step=step)
                self.mlflow_tracker.log_metric("latency_ms", latency_ms, step=step)
                
                # J. Visualization Updates (Prometheus gauges + Terminal Dashboard render)
                self.dashboard_exporter.update(
                    latency_ms=latency_ms,
                    throughput=throughput,
                    score=score_norm,
                    threshold=threshold,
                    alarm=is_alarm,
                    condition_number=cond,
                    singular_values=S,
                    event_count=event_count
                )
                
                self.dashboard_exporter.render_console_dashboard(
                    step=step,
                    score=score_norm,
                    threshold=threshold,
                    alarm=is_alarm,
                    condition_number=cond,
                    singular_values=S,
                    latency_ms=latency_ms,
                    throughput=throughput
                )
                
                step += 1
                time.sleep(1.0)  # Periodic pipeline aggregation interval
                
        except KeyboardInterrupt:
            logger.info("Pipeline execution paused by user request.")
        finally:
            self.stop()

    def stop(self) -> None:
        """
        Gracefully releases system resources.
        """
        self.is_running = False
        self.consumer.close()
        self.redis_manager.close()
        self.mlflow_tracker.end_run()
        logger.info("Streaming pipeline shut down successfully.")

if __name__ == "__main__":
    manager = StreamingPipelineManager()
    manager.run()
