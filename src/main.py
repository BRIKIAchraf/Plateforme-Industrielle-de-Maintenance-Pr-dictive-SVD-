import time
import sys
import os
import yaml
import numpy as np
import redis

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing.imputation import MissingDataEngine
from src.signal.processing import IndustrialSignalProcessor
from src.models.incremental_svd import OnlineIncrementalSVD
from src.models.autoencoder import AutoencoderEngine
from src.anomaly.hybrid_engine import HybridAnomalyEngine
from src.evaluation.phm_metrics import PHMValidationEngine
from src.monitoring.metrics_exporter import PrometheusMetricsExporter

import re

def resolve_env_vars(data):
    """
    Recursively replaces environmental variable placeholders like ${VAR:-DEFAULT} or ${VAR}
    with actual system environment variables or their default values, casting numeric types.
    """
    if isinstance(data, dict):
        return {k: resolve_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [resolve_env_vars(item) for item in data]
    elif isinstance(data, str):
        # Match pattern ${VAR:-DEFAULT} or ${VAR}
        match = re.match(r'^\$\{([^:]+)(?::-([^}]+))?\}$', data)
        if match:
            var_name, default_val = match.groups()
            val = os.environ.get(var_name, default_val if default_val is not None else "")
        else:
            val = data
            
        # Try to cast to float/int if possible (handles placeholders and raw strings like 1e-4)
        try:
            if '.' in val or 'e-' in val or 'e+' in val:
                return float(val)
            return int(val)
        except ValueError:
            # If casting fails, handle boolean values
            if val.lower() == 'true': return True
            if val.lower() == 'false': return False
            return val
    return data

def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, 'r') as f:
        raw_data = yaml.safe_load(f)
        return resolve_env_vars(raw_data)

def run_pipeline():
    print("=" * 80)
    print("      LAUNCHING INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM (TOP TIER)")
    print("=" * 80)
    
    # 1. Load Configurations
    config = load_config()
    n_components = config['models']['incremental_svd']['n_components']
    forgetting_factor = config['models']['incremental_svd']['forgetting_factor']
    ewma_alpha = config['models']['hybrid_anomaly']['ewma_alpha']
    initial_k = config['models']['hybrid_anomaly']['initial_threshold']
    
    # 2. Initialize scientific and engineering layers
    print("[1/6] Initializing Imputation and Signal Processing Layers...")
    imputation_engine = MissingDataEngine(
        kalman_q=config['preprocessing']['imputation']['kalman_q'],
        kalman_r=config['preprocessing']['imputation']['kalman_r']
    )
    signal_processor = IndustrialSignalProcessor(
        sampling_rate_hz=config['signal_processing']['sampling_rate_hz']
    )

    print("[2/6] Building Hybrid AI Models (Online SVD & PyTorch Autoencoder)...")
    n_sensors = config['models']['autoencoder']['input_dim']
    online_svd = OnlineIncrementalSVD(n_components=n_components, forgetting_factor=forgetting_factor)
    ae_engine = AutoencoderEngine(input_dim=n_sensors, latent_dim=config['models']['autoencoder']['latent_dim'])
    
    print("[3/6] Setting Up Multicore Anomaly Detection Engine...")
    anomaly_engine = HybridAnomalyEngine(ewma_alpha=ewma_alpha, initial_threshold=initial_k)
    phm_engine = PHMValidationEngine(failure_threshold=6.5)

    print("[4/6] Connecting to Redis Key-Value Store...")
    try:
        r_client = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            password=config['redis']['password'],
            decode_responses=True
        )
        r_client.ping()
        print(" -> Redis successfully connected.")
    except Exception as e:
        print(f" -> Redis connection omitted (local simulation active): {e}")
        r_client = None

    print("[5/6] Launching Prometheus Telemetry Exporter...")
    exporter = PrometheusMetricsExporter(port=config['prometheus']['port'])
    exporter.start()

    print("[6/6] Initializing Warm-Up State...")
    # Warmup with standard nominal sensor vibrations
    warmup_samples = 200
    nominal_baseline = np.random.normal(0.0, 0.5, (warmup_samples, n_sensors))
    online_svd.update(nominal_baseline)
    ae_engine.train_step(nominal_baseline, epochs=5, batch_size=32)
    anomaly_engine.update_mahalanobis_covariance(nominal_baseline)
    
    print("=" * 80)
    print("STREAMING ACTIVE (Press Ctrl+C to terminate)")
    print("-" * 80)
    
    score_history = []
    step = 0
    
    try:
        while True:
            t_start = time.perf_counter()
            
            # Simulate a live window of incoming sensor readings (100 samples)
            # Inject rising vibration (progressive drift) after step 15 to trigger an alarm
            drift_coeff = max(0.0, (step - 15) * 0.08)
            raw_vibrations = np.random.normal(0.0, 0.4 + drift_coeff, (100, n_sensors))
            
            # Inject a sudden sensor blackout/missing reading (NaN spike)
            if step == 5:
                raw_vibrations[10:15, 2] = np.nan
            
            # --- Pipeline execution ---
            # A. Impute & Smooth (Kalman Filtering)
            cleaned_vibrations = imputation_engine.clean_sensor_matrix(raw_vibrations)
            
            # B. Signal Preprocessing (Detrend, Filter, Wavelets)
            processed_vibrations = signal_processor.preprocess_pipeline(
                cleaned_vibrations, 
                lowpass_cutoff=config['signal_processing']['butterworth']['cutoff_hz']
            )
            
            # C. Compute FFT for frequency domain entropy
            freqs, amplitudes = signal_processor.compute_fft(processed_vibrations)
            freq_entropy = anomaly_engine.calculate_frequency_entropy(amplitudes)
            
            # D. Model Updates & Projections
            online_svd.update(processed_vibrations)
            svd_errors = online_svd.get_reconstruction_error(processed_vibrations)
            svd_res = float(np.mean(svd_errors))
            spectral_drift = online_svd.get_spectral_drift()
            
            # E. PyTorch Autoencoder Evaluation
            ae_errors = ae_engine.evaluate_reconstruction_error(processed_vibrations)
            ae_res = float(np.mean(ae_errors))
            
            # F. Mahalanobis distance update & extraction
            mean_obs = np.mean(processed_vibrations, axis=0)
            mahalanobis_dist = anomaly_engine.calculate_mahalanobis_distance(mean_obs)
            
            # G. Score Fusion & Adaptive EWMA alerts
            fused_score = anomaly_engine.fuse_anomaly_scores(
                svd_residual=svd_res,
                ae_residual=ae_res,
                mahalanobis_dist=mahalanobis_dist,
                freq_entropy=freq_entropy,
                spectral_drift=spectral_drift
            )
            
            score_history.append(fused_score)
            
            threshold, is_alarm = anomaly_engine.update_ewma_threshold(fused_score)
            
            # H. Extrapolate Remaining Useful Life (RUL)
            rul = phm_engine.estimate_remaining_useful_life(score_history, time_step_seconds=1.0)
            
            t_end = time.perf_counter()
            latency_ms = (t_end - t_start) * 1000
            throughput = 100 / (t_end - t_start)
            
            # I. Publish metrics to Prometheus
            exporter.update_metrics(
                latency_ms=latency_ms,
                throughput=throughput,
                score=fused_score,
                threshold=threshold,
                alarm=is_alarm,
                drift=spectral_drift,
                rul=rul if rul != float('inf') else 9999.0,
                event_count=100
            )
            
            # J. Push alert event log to Redis
            if is_alarm and r_client:
                r_client.hset(f"alarm:{step}", mapping={
                    "timestamp": time.time(),
                    "score": fused_score,
                    "threshold": threshold,
                    "estimated_rul": rul
                })
                r_client.expire(f"alarm:{step}", 3600)  # Expire after 1 hour

            # Print telemetry console log
            alarm_str = "\033[91m⚠️ ALARM TRIGGERED\033[0m" if is_alarm else "\033[92mNominal\033[0m"
            rul_str = f"{rul:.1f}s" if rul != float('inf') else "Stable"
            
            print(f"Step {step:02d} | Latency: {latency_ms:.2f}ms | Score: {fused_score:.3f} (Thresh: {threshold:.3f}) | Status: {alarm_str:<25} | RUL: {rul_str}")
            
            step += 1
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nPipeline gracefully shut down by user request.")

if __name__ == "__main__":
    run_pipeline()
