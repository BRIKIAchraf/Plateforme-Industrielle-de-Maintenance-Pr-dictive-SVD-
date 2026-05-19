import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.kafka_consumer import KafkaSensorConsumer
from src.ingestion.stream_buffer import RealTimeStreamBuffer
from src.anomaly.thresholding import AdaptiveEWMAThresholding

def test_mock_kafka_consumer_polling():
    # Initialize in mock mode
    consumer = KafkaSensorConsumer(mock=True)
    messages = list(consumer.poll_messages())
    
    assert len(messages) == 100  # We simulate 100 sensors per step
    for msg in messages:
        assert "sensor_id" in msg
        assert "timestamp" in msg
        assert "vibration" in msg
        assert "temperature" in msg

def test_stream_buffer_matrix_alignment():
    # Size 5 steps
    buffer = RealTimeStreamBuffer(window_size_seconds=5.0, max_time_steps=5)
    
    # Push out of order data for sensor 0 and 1
    buffer.append(sensor_id=0, timestamp=100.0, vibration=0.1, temperature=40.0)
    buffer.append(sensor_id=0, timestamp=101.0, vibration=0.2, temperature=40.5)
    buffer.append(sensor_id=0, timestamp=102.0, vibration=0.3, temperature=41.0)
    
    buffer.append(sensor_id=1, timestamp=100.2, vibration=0.5, temperature=50.0)
    buffer.append(sensor_id=1, timestamp=101.5, vibration=0.6, temperature=50.2)
    
    X, sensor_ids = buffer.get_feature_matrix()
    
    assert len(sensor_ids) == 2
    assert sensor_ids == [0, 1]
    # Matrix shape should be (n_sensors, max_time_steps) -> (2, 5)
    assert X.shape == (2, 5)
    # Check that there are no NaNs in the final feature matrix
    assert not np.isnan(X).any()

def test_stream_buffer_interpolation_nans():
    buffer = RealTimeStreamBuffer(window_size_seconds=5.0, max_time_steps=3)
    
    # Sensor 0 with NaN in the middle
    buffer.append(sensor_id=0, timestamp=10.0, vibration=1.0, temperature=50.0)
    buffer.append(sensor_id=0, timestamp=11.0, vibration=np.nan, temperature=50.0)
    buffer.append(sensor_id=0, timestamp=12.0, vibration=3.0, temperature=50.0)
    
    X, sensor_ids = buffer.get_feature_matrix()
    
    # Check that NaN was correctly interpolated (should be around 2.0 at index 1)
    assert not np.isnan(X).any()
    assert np.isclose(X[0, 1], 2.0, atol=0.1)

def test_adaptive_threshold_triggers():
    # Adaptive thresholding
    detector = AdaptiveEWMAThresholding(alpha=0.2, k=3.0)
    
    # Feed stable scores
    for _ in range(20):
        detector.update(1.0)
        
    # Feed a massive spike to trigger anomaly
    thresh, alarm = detector.update(10.0)
    
    assert alarm is True
    assert thresh < 10.0
