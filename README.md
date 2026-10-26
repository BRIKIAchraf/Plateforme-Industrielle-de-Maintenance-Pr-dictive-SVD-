# Predictive Maintenance Platform with SVD

Real-time anomaly-detection architecture for industrial sensor streams using low-rank reconstruction, adaptive thresholds and MLOps tracking.

## Problem

Industrial machines generate correlated sensor signals. A change in the normal low-rank structure can indicate degradation or an emerging fault. This project uses Singular Value Decomposition (SVD) to model the normal state and calculate reconstruction residuals.

## Data flow

~~~text
Sensor events → Kafka consumer → stream buffer → feature matrix X
                                      ↓
                         truncated / randomized SVD
                                      ↓
                 reconstruction X-hat → residual score
                                      ↓
                    EWMA threshold → alert + persisted state
~~~

## Detection method

For a feature matrix (X), the low-rank approximation is:

~~~text
X-hat = U-k × Sigma-k × V-k-transpose
~~~

The normalized Frobenius residual provides the anomaly score:

~~~text
score = ||X - X-hat||F / ||X||F
~~~

The pipeline also tracks numerical conditioning and supports:

- static baseline thresholds;
- adaptive EWMA alerting;
- Sherman–Morrison updates for changing correlations;
- Redis state for current machine status and alarm history;
- MLflow experiment and parameter tracking.

## Architecture

| Component | Responsibility |
|---|---|
| Kafka | Sensor event transport and replay |
| Stream buffer | Align asynchronous sensor signals |
| SVD modules | Low-rank modelling and reconstruction |
| Anomaly modules | Residual score and thresholding |
| Redis | Live state and alert persistence |
| MLflow | Experiment tracking |
| Prometheus/Grafana | Metrics and operational visibility |

## Project structure

~~~text
src/ingestion/       Kafka consumer and stream buffer
src/math/            SVD, norms and numerical stability
src/anomaly/         Reconstruction, scores and thresholds
src/pipeline/        Streaming and batch orchestration
src/storage/         Redis and MLflow integrations
src/visualization/   Metrics exposure
tests/               Mathematical and streaming tests
docker/              Runtime configuration
~~~

## Running locally

~~~bash
docker compose up -d
pip install -r requirements.txt
~~~

Run the relevant simulator, streaming service and tests according to the repository configuration.

## Important limitations

This is a portfolio/research implementation. A real predictive-maintenance deployment needs labelled failure history, threshold calibration, false-positive analysis, sensor-quality controls, model validation, alert ownership and an explicit RUL evaluation protocol. The capacity claims must be validated with representative load tests before production use.

## Author

**Achraf BRIKI** · [LinkedIn](https://www.linkedin.com/in/achrafbriki/) · [Portfolio](https://achrafbriki.com)
