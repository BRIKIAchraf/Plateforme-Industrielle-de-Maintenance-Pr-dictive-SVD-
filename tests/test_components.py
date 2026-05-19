import pytest
import numpy as np
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing.imputation import KalmanFilter1D, MissingDataEngine
from src.signal.processing import IndustrialSignalProcessor
from src.models.incremental_svd import OnlineIncrementalSVD
from src.models.autoencoder import AutoencoderEngine
from src.anomaly.hybrid_engine import HybridAnomalyEngine
from src.evaluation.phm_metrics import PHMValidationEngine

def test_kalman_filter_and_imputation():
    # Test 1: 1D Kalman Filter
    kf = KalmanFilter1D(q=1e-4, r=1e-2)
    # Feed clean sequence
    for val in [1.0, 1.0, 1.0, 1.0]:
        kf.update(val)
    # Feed missing measurement (NaN)
    imputed_val = kf.update(np.nan)
    assert np.isclose(imputed_val, 1.0, atol=0.1)

    # Test 2: Missing Data Engine
    engine = MissingDataEngine()
    raw_matrix = np.array([
        [1.0, 2.0],
        [np.nan, 2.1],
        [1.2, np.nan],
        [1.1, 2.2]
    ])
    cleaned = engine.clean_sensor_matrix(raw_matrix)
    assert not np.isnan(cleaned).any()
    assert cleaned.shape == (4, 2)

def test_signal_processor():
    processor = IndustrialSignalProcessor(sampling_rate_hz=1000.0)
    
    # Generate 1 second of 50 Hz sine wave + high frequency noise (300 Hz)
    t = np.linspace(0, 1, 1000)
    clean_signal = np.sin(2 * np.pi * 50 * t)
    noisy_signal = clean_signal + 0.5 * np.sin(2 * np.pi * 300 * t)
    
    # Expand to 2D for processing along columns
    data = noisy_signal.reshape(-1, 1)
    
    # Apply Butterworth lowpass
    filtered = processor.butterworth_lowpass(data, cutoff_hz=100.0)
    # Apply Haar wavelet denoising
    denoised = processor.haar_wavelet_denoise(data, level=2)
    
    assert filtered.shape == (1000, 1)
    assert denoised.shape == (1000, 1)
    # The high frequency noise should be reduced in the filtered signal
    assert np.var(filtered - clean_signal.reshape(-1, 1)) < np.var(data - clean_signal.reshape(-1, 1))

def test_incremental_svd():
    svd = OnlineIncrementalSVD(n_components=2, forgetting_factor=0.95)
    
    # Generate matrix
    X1 = np.random.normal(0, 1, (100, 4))
    X2 = np.random.normal(0, 1, (100, 4))
    
    svd.update(X1)
    svd.update(X2)
    
    reconstructed = svd.reconstruct(X2)
    errors = svd.get_reconstruction_error(X2)
    
    assert reconstructed.shape == (100, 4)
    assert errors.shape == (100,)
    assert svd.get_spectral_drift() > 0.0

def test_autoencoder_engine():
    ae = AutoencoderEngine(input_dim=5, latent_dim=2)
    
    X = np.random.normal(0, 1, (50, 5))
    initial_loss = ae.train_step(X, epochs=3, batch_size=10)
    
    errors = ae.evaluate_reconstruction_error(X)
    latent = ae.get_latent_representation(X)
    
    assert errors.shape == (50,)
    assert latent.shape == (50, 2)
    assert initial_loss > 0.0

def test_hybrid_anomaly_engine():
    engine = HybridAnomalyEngine(ewma_alpha=0.2, initial_threshold=3.0)
    
    # Setup covariance
    X_train = np.random.normal(0, 1, (100, 3))
    engine.update_mahalanobis_covariance(X_train)
    
    # Compute scores
    dist = engine.calculate_mahalanobis_distance(np.array([0.5, 0.5, 0.5]))
    assert dist >= 0.0
    
    fused = engine.fuse_anomaly_scores(
        svd_residual=1.2,
        ae_residual=0.8,
        mahalanobis_dist=dist,
        freq_entropy=0.5,
        spectral_drift=0.1
    )
    assert fused > 0
    
    threshold, is_anomaly = engine.update_ewma_threshold(fused)
    assert threshold > 0.0
    assert isinstance(is_anomaly, bool)

def test_phm_validation_engine():
    engine = PHMValidationEngine(failure_threshold=10.0)
    
    # Stable degradation score history starting at 1.0 and ramping up to 8.0
    scores = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    
    rul = engine.estimate_remaining_useful_life(scores, time_step_seconds=1.0)
    # Slope is 0.5 units per step. Distance remaining: 10.0 - 8.0 = 2.0 units.
    # Expected RUL: 2.0 / 0.5 = 4.0 seconds.
    assert np.isclose(rul, 4.0, atol=0.1)
    
    # Alarm metrics
    alerts = np.array([False, False, True, True, True])
    ground_truth = np.array([False, False, False, True, True])
    
    metrics = engine.calculate_alarm_performance(alerts, ground_truth)
    assert metrics["false_alarm_rate"] == pytest.approx(1.0 / 3.0) # 1 false alarm out of 3 normal steps
    assert metrics["missed_detection_rate"] == pytest.approx(0.0) # both failures detected
    
    delay = engine.compute_early_detection_delay(alerts, failure_index=3, time_step_seconds=2.0)
    # failure index is 3 (ground truth fails at index 3 and 4)
    # first stable alert index is 2
    # Delay = (3 - 2) * 2.0 = 2.0 seconds
    assert delay == 2.0
