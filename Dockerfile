FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /build

# Copy requirements
COPY backend/requirements.txt .

# Install Python dependencies into a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.9-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    libsdl2-2.0-0 \
    libsdl2-image-2.0-0 \
    xvfb \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Set up working directory
WORKDIR /app

# Copy backend code
COPY backend/ /app/

# Copy game assets (only what's needed)
COPY imgs/ /app/imgs/

# Create necessary directories with minimal permissions
RUN mkdir -p /app/models /app/data /app/logs && \
    chmod -R 755 /app/models /app/data /app/logs

# Set environment variables for Pygame in headless mode
ENV SDL_VIDEODRIVER=dummy
ENV DISPLAY=:99
ENV PYTHONUNBUFFERED=1

# Start Xvfb and run the application
CMD Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset & \
    python app.py 