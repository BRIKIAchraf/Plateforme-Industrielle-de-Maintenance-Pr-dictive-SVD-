# Kafka Deployment Configuration

This directory contains configuration files for the Apache Kafka message broker used in our real-time streaming pipeline.

## Configured Topics
- `machinery_sensors`: The topic used to ingest real-time vibration and temperature data from the 10,000+ IoT sensors.

## Docker compose service:
The Kafka and Zookeeper cluster are orchestrated using the root `docker-compose.yml` file, running CP-Kafka image version `7.3.0`.
