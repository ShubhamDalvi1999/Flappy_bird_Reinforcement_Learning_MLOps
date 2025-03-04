FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for pygame and mlflow
RUN apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    build-essential \
    git \
    curl \
    default-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/

# Create directories for models, data, logs, and ensure proper permissions
RUN mkdir -p /app/models /app/data /app/logs /tmp/gunicorn && \
    chmod -R 777 /app/models /app/data /app/logs /tmp/gunicorn

# Make port 5000 available for the app
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV DOCKER_ENV=1
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV GUNICORN_WORKERS=2
ENV GUNICORN_THREADS=4
ENV GUNICORN_TIMEOUT=120

# Run app with gunicorn
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 --workers ${GUNICORN_WORKERS} --threads ${GUNICORN_THREADS} --timeout ${GUNICORN_TIMEOUT} --worker-tmp-dir /tmp/gunicorn app:app"] 