FROM python:3.9-slim-buster

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Java JDK (required by PySpark execution) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY configs/ ./configs/
COPY src/ ./src/
COPY tests/ ./tests/

# Expose the Prometheus metrics port
EXPOSE 8000

# Set entrypoint to run our streaming pipeline
CMD ["python", "src/pipeline/streaming_pipeline.py"]
