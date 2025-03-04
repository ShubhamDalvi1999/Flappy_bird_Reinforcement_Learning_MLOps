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
    openjdk-17-jre-headless \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ /app/

# Create directories for models, data, and logs
RUN mkdir -p /app/models /app/data /app/logs

# Make port 5000 available for the app
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://localhost:5000

# Run app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"] 