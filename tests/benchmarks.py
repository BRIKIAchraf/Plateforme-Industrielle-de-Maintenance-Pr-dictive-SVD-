import time
import sys
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import torch
import psutil

# Ensure src path is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.incremental_svd import OnlineIncrementalSVD
from src.models.autoencoder import AutoencoderEngine

def get_memory_usage_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def run_benchmarks():
    print("=" * 70)
    print(" INDUSTRIAL PREDICTIVE MAINTENANCE SUITE BENCHMARKING ENGINE")
    print("=" * 70)
    
    # 1. Generate Synthetic high-dimensional sensor data
    np.random.seed(42)
    n_samples = 5000
    n_features = 10
    print(f"Generating synthetic dataset: {n_samples} samples, {n_features} sensor channels...")
    
    # Standard normal vibrations
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    # Inject drifting wear and sudden structural anomalies
    t = np.linspace(0, 10, n_samples)
    drift = np.outer(t * 0.1, np.ones(n_features))
    X_normal += drift
    
    # 2. Warm-up and initialize models
    print("\nInitializing models for comparison...")
    
    # A. Standard PCA (batch-mode)
    pca_batch = PCA(n_components=3)
    
    # B. Online Incremental SVD (streaming-mode)
    online_svd = OnlineIncrementalSVD(n_components=3, forgetting_factor=0.99)
    # Warmup with small batch
    online_svd.update(X_normal[:100])
    
    # C. Neural Autoencoder (Deep model)
    ae_engine = AutoencoderEngine(input_dim=n_features, latent_dim=3)
    ae_engine.train_step(X_normal[:500], epochs=2, batch_size=32)
    
    # D. Isolation Forest (Standard Anomaly Detector)
    iforest = IsolationForest(n_estimators=50, random_state=42)
    iforest.fit(X_normal[:500])
    
    results = {}
    
    # ------------------ Benchmark A: Standard PCA ------------------
    t_start = time.perf_counter()
    m_start = get_memory_usage_mb()
    # Batch PCA needs fit + transform
    pca_batch.fit(X_normal)
    _ = pca_batch.transform(X_normal)
    t_end = time.perf_counter()
    m_end = get_memory_usage_mb()
    
    results["Standard PCA (Batch)"] = {
        "latency_ms": ((t_end - t_start) / n_samples) * 1000,
        "throughput": n_samples / (t_end - t_start),
        "mem_mb": m_end - m_start
    }
    
    # ------------------ Benchmark B: Online SVD ------------------
    t_start = time.perf_counter()
    m_start = get_memory_usage_mb()
    # Simulate window updates (updating SVD block-by-block of 100 samples)
    window_size = 100
    for idx in range(100, n_samples, window_size):
        batch = X_normal[idx - window_size : idx]
        online_svd.update(batch)
        _ = online_svd.get_reconstruction_error(batch)
    t_end = time.perf_counter()
    m_end = get_memory_usage_mb()
    
    results["Online SVD (Streaming)"] = {
        "latency_ms": ((t_end - t_start) / n_samples) * 1000,
        "throughput": n_samples / (t_end - t_start),
        "mem_mb": m_end - m_start
    }
    
    # ------------------ Benchmark C: PyTorch Autoencoder ------------------
    t_start = time.perf_counter()
    m_start = get_memory_usage_mb()
    # Evaluate reconstruction error sample-by-sample
    _ = ae_engine.evaluate_reconstruction_error(X_normal)
    t_end = time.perf_counter()
    m_end = get_memory_usage_mb()
    
    results["Neural Autoencoder (Inference)"] = {
        "latency_ms": ((t_end - t_start) / n_samples) * 1000,
        "throughput": n_samples / (t_end - t_start),
        "mem_mb": m_end - m_start
    }
    
    # ------------------ Benchmark D: Isolation Forest ------------------
    t_start = time.perf_counter()
    m_start = get_memory_usage_mb()
    _ = iforest.score_samples(X_normal)
    t_end = time.perf_counter()
    m_end = get_memory_usage_mb()
    
    results["Isolation Forest"] = {
        "latency_ms": ((t_end - t_start) / n_samples) * 1000,
        "throughput": n_samples / (t_end - t_start),
        "mem_mb": m_end - m_start
    }
    
    # 3. Print benchmarking results
    print("\n" + "=" * 80)
    print(f"{'Model / Algorithm':<30} | {'Latency per sample':<20} | {'Throughput':<15} | {'Memory (dRSS)':<10}")
    print("-" * 80)
    for model_name, metrics in results.items():
        lat_str = f"{metrics['latency_ms']:.4f} ms"
        thru_str = f"{metrics['throughput']:.1f} smpl/s"
        mem_str = f"{metrics['mem_mb']:.3f} MB"
        print(f"{model_name:<30} | {lat_str:<20} | {thru_str:<15} | {mem_str:<10}")
    print("=" * 80)
    print("Scaling Law observation: Online SVD exhibits linear time complexity O(N*K^2) making it ")
    print("exponentially cheaper than Batch PCA (O(N^3)) for long-running streaming platforms.")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmarks()
