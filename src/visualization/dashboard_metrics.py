import os
import sys
import time
import logging
import numpy as np
from typing import Dict, Any, List

try:
    from prometheus_client import start_http_server, Gauge, Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger("DashboardMetrics")

class DashboardMetricsExporter:
    """
    Exposes industrial telemetry metrics to Prometheus for Grafana charts, 
    and drives a real-time high-fidelity terminal dashboard for instant visibility.
    """
    def __init__(self, port: int = 8000):
        self.port = port
        self.prometheus_active = False
        
        if PROMETHEUS_AVAILABLE:
            try:
                # Metric Definitions
                self.processing_latency = Gauge(
                    'svd_pipeline_processing_latency_ms_custom', 
                    'Time taken to preprocess and perform SVD reconstruction'
                )
                self.event_throughput = Gauge(
                    'svd_pipeline_throughput_events_per_sec_custom', 
                    'Number of incoming sensor events processed per second'
                )
                self.anomaly_score = Gauge(
                    'svd_pipeline_residual_anomaly_score', 
                    'Unified normalized Frobenius anomaly score'
                )
                self.adaptive_threshold = Gauge(
                    'svd_pipeline_dynamic_threshold_custom', 
                    'Current EWMA dynamic alert boundary'
                )
                self.alarm_state = Gauge(
                    'svd_pipeline_alarm_triggered_custom', 
                    'Binary indicator representing active machine failure warning (1=alarm, 0=nominal)'
                )
                self.condition_number = Gauge(
                    'svd_pipeline_condition_number', 
                    'Matrix condition number sigma_max / sigma_min'
                )
                self.total_processed_events = Counter(
                    'svd_pipeline_processed_events_total_custom', 
                    'Total cumulative number of sensor events ingested'
                )
                
                start_http_server(self.port)
                self.prometheus_active = True
                logger.info(f"Prometheus HTTP Exporter started on port {self.port}")
            except Exception as e:
                logger.warning(f"Could not start Prometheus HTTP server: {e}. Grafana exports disabled.")
        else:
            logger.warning("prometheus_client package is not installed. Grafana exports disabled.")

    def update(
        self, 
        latency_ms: float, 
        throughput: float, 
        score: float, 
        threshold: float, 
        alarm: bool, 
        condition_number: float, 
        singular_values: np.ndarray,
        event_count: int
    ) -> None:
        """
        Pushes latest telemetry data to Prometheus scrapers.
        """
        if self.prometheus_active:
            try:
                self.processing_latency.set(latency_ms)
                self.event_throughput.set(throughput)
                self.anomaly_score.set(score)
                self.adaptive_threshold.set(threshold)
                self.alarm_state.set(1.0 if alarm else 0.0)
                self.condition_number.set(condition_number)
                self.total_processed_events.inc(event_count)
            except Exception as e:
                logger.error(f"Prometheus metric update failed: {e}")

    def render_console_dashboard(
        self, 
        step: int, 
        score: float, 
        threshold: float, 
        alarm: bool, 
        condition_number: float, 
        singular_values: np.ndarray, 
        latency_ms: float, 
        throughput: float
    ) -> None:
        """
        Renders a stunning terminal-based real-time dashboard representing 
        the state-of-the-art predictive maintenance pipeline metrics.
        """
        # Clear screen for live updates
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        alarm_status = "\033[91m⚠️ [ALARM: MECHANICAL ANOMALY DETECTED]\033[0m" if alarm else "\033[92m🟢 [NOMINAL: MACHINERY STABLE]\033[0m"
        
        stability_color = "\033[92m"  # Green
        if condition_number >= 1000.0:
            stability_color = "\033[91m"  # Red (numerical instability)
        elif condition_number >= 100.0:
            stability_color = "\033[93m"  # Yellow
            
        print("\n" + "=" * 90)
        print(" " * 22 + "🔥 INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM (SVD) 🔥")
        print("=" * 90)
        print(f" Pipeline Step: {step:<6} | Real-Time Latency: {latency_ms:.2f} ms | Throughput: {throughput:.1f} events/s")
        print(f" Status       : {alarm_status}")
        print("-" * 90)
        
        # Draw Anomaly Score visual progress bar
        bar_len = 30
        filled = int(min(1.0, score / max(1e-5, threshold * 2.0)) * bar_len)
        bar = "█" * filled + "-" * (bar_len - filled)
        print(f" Anomaly Score     : {score:.4f} / Thresh: {threshold:.4f} | [{bar}]")
        
        # Singular Values Spectrum ASCII plot
        print(" Singular Spectrum : ", end="")
        if len(singular_values) > 0:
            norm_sv = singular_values / (np.sum(singular_values) + 1e-9)
            spectrum_bars = []
            for s in norm_sv:
                heights = [" ", "_", "▃", "▅", "▇", "█"]
                idx = min(len(heights) - 1, int(s * 10))
                spectrum_bars.append(heights[idx])
            print("".join(spectrum_bars) + f"  (k_energy={len(singular_values)} components)")
        else:
            print("No singular spectrum data.")
            
        # Numerical Stability
        print(f" Condition Number  : {stability_color}{condition_number:.2f}\033[0m", end="")
        if condition_number < 100.0:
            print(f" (Numerical Matrix is \033[92mSTABLE\033[0m)")
        elif condition_number < 1000.0:
            print(f" (Numerical Matrix is \033[93mMODERATE\033[0m - monitor drift)")
        else:
            print(f" (Numerical Matrix is \033[91mUNSTABLE\033[0m - high risk of singular matrix!)")
            
        print("=" * 90 + "\n")
        sys.stdout.flush()
