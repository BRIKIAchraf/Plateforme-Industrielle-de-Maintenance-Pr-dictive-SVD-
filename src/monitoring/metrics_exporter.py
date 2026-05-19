from prometheus_client import start_http_server, Gauge, Counter
import time
import logging

class PrometheusMetricsExporter:
    """
    Exposes industrial telemetry and MLOps metrics to Prometheus.
    """
    def __init__(self, port: int = 8000):
        self.port = port
        
        # 1. Define Prometheus Gauges & Counters
        self.processing_latency = Gauge(
            'svd_pipeline_processing_latency_ms', 
            'Time taken to preprocess and perform SVD/AE reconstruction'
        )
        self.event_throughput = Gauge(
            'svd_pipeline_throughput_events_per_sec', 
            'Number of incoming sensor events processed per second'
        )
        self.anomaly_score = Gauge(
            'svd_pipeline_fused_anomaly_score', 
            'Unified Multi-Score Health Index'
        )
        self.adaptive_threshold = Gauge(
            'svd_pipeline_dynamic_threshold', 
            'Current EWMA dynamic alert boundary'
        )
        self.alarm_state = Gauge(
            'svd_pipeline_alarm_triggered', 
            'Binary indicator representing active machine failure warning (1=alarm, 0=nominal)'
        )
        self.spectral_drift = Gauge(
            'svd_pipeline_spectral_drift', 
            'Long-term singular spectrum mechanical degradation index'
        )
        self.estimated_rul = Gauge(
            'svd_pipeline_estimated_rul_seconds', 
            'Extrapolated Remaining Useful Life of the machinery in seconds'
        )
        self.total_processed_events = Counter(
            'svd_pipeline_processed_events_total', 
            'Total cumulative number of sensor events ingested'
        )

    def start(self) -> None:
        """
        Launch the Prometheus scrapable HTTP server.
        """
        try:
            start_http_server(self.port)
            logging.info(f"Prometheus HTTP Exporter started on port {self.port}")
            print(f"Prometheus HTTP Exporter running on port {self.port}")
        except Exception as e:
            logging.error(f"Failed to start Prometheus Exporter: {e}")
            print(f"Prometheus Exporter port {self.port} already running or unavailable: {e}")

    def update_metrics(
        self, 
        latency_ms: float, 
        throughput: float, 
        score: float, 
        threshold: float, 
        alarm: bool, 
        drift: float, 
        rul: float,
        event_count: int
    ) -> None:
        """
        Dynamically updates the Prometheus gauge metrics.
        """
        self.processing_latency.set(latency_ms)
        self.event_throughput.set(throughput)
        self.anomaly_score.set(score)
        self.adaptive_threshold.set(threshold)
        self.alarm_state.set(1.0 if alarm else 0.0)
        self.spectral_drift.set(drift)
        self.estimated_rul.set(rul)
        self.total_processed_events.inc(event_count)
