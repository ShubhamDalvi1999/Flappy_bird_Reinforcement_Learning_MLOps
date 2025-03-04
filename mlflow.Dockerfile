FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /build

# Create a virtual environment and install MLflow
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir 'mlflow==2.10.0' 'sqlalchemy==2.0.23' psutil

# Final stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl sqlite3 && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Set up working directory and create directories
WORKDIR /mlflow
RUN mkdir -p /mlflow/artifacts

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_HOME=/mlflow
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts
ENV MLFLOW_SERVE_ARTIFACTS=true

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=30s \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Start MLflow server
CMD if [ ! -f /mlflow/mlflow.db ]; then \
        echo 'Initializing MLflow database...' && \
        mlflow db init sqlite:///mlflow/mlflow.db && \
        echo 'Database initialization complete.'; \
    fi && \
    echo 'Starting MLflow server...' && \
    mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow/mlflow.db \
    --default-artifact-root /mlflow/artifacts \
    --serve-artifacts 