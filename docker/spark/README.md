# Spark Streaming Configuration

This directory contains configuration guidelines and memory tuning properties for PySpark structured streaming.

## Processing Metrics
- **Watermark Duration**: 10 seconds (allows late sensor data to be aggregated without causing out-of-memory issues).
- **Trigger Processing Interval**: 5 seconds (runs micro-batch operations to feed SVD models).
- **Backpressure**: Enabled (`initialRate: 1000`, `maxRate: 10000`) to dynamically control ingestion rate under heavy bursts.
