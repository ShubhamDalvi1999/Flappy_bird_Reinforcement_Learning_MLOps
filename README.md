# Flappy Bird Reinforcement Learning (Production Grade)

A production-grade AI/ML application that uses Deep Q-Learning to train an agent to play Flappy Bird. This project demonstrates best practices for developing, deploying, and monitoring ML applications in production.

## Features

- **Deep Q-Learning Agent**: A neural network that learns to play Flappy Bird
- **Experiment Tracking**: MLflow integration for tracking experiments and model versions
- **Observability**: Weights & Biases integration for real-time visualizations and alerting
- **Scalable Data Processing**: PySpark integration for data analysis and preprocessing
- **Testing Framework**: Pytest integration for testing models and environment
- **Containerization**: Docker and Docker Compose for easy deployment
- **Web Interface**: Interactive Next.js frontend for visualization and control

## Architecture

The application consists of several components:

- **Backend (Python/Flask)**: Handles training, inference, and API endpoints
- **Frontend (Next.js)**: Provides visualization and user controls
- **MLflow Server**: Manages experiment tracking and model registry

## Quick Start

### Prerequisites

- Docker and Docker Compose
- (Optional) NVIDIA GPU with CUDA support for faster training

### Running with Docker

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/flappy-bird-rl.git
cd flappy-bird-rl
```

2. **Set up Weights & Biases (Optional but recommended)**

On Windows:
```bash
.\setup-wandb.bat
```

On Unix/Linux:
```bash
export WANDB_API_KEY=your_api_key_here
```

3. **Run the application**

On Windows:
```bash
.\run-docker.bat
```

On Unix/Linux:
```bash
./run-docker.sh
```

4. **Access the application**

- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- MLflow UI: http://localhost:5001

### GPU Support

To check if your system supports GPU acceleration:

On Windows:
```bash
.\check-gpu.bat
```

On Unix/Linux:
```bash
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

If GPU is available, you can enable it by setting:
```
USE_GPU=Dockerfile.gpu
GPU_COUNT=1
```
in your `.env` file.

### Simplified Setup

For a minimal setup without GPU support:

On Windows:
```bash
.\run-simple.bat
```

On Unix/Linux:
```bash
docker-compose -f docker-compose.simple.yml up -d
```

## Project Structure

```
flappy-bird-rl/
├── backend/                # Python backend
│   ├── agent/              # DQN agent implementation
│   ├── game/               # Flappy Bird environment
│   ├── training/           # Training logic
│   ├── tests/              # Unit tests
│   └── app.py              # Flask API
├── frontend/               # Next.js frontend
│   ├── components/         # React components
│   ├── pages/              # Next.js pages
│   └── public/             # Static assets
├── docker-compose.yml      # Main Docker Compose configuration
├── docker-compose.simple.yml # Simplified Docker Compose
├── Dockerfile              # Backend Dockerfile
├── Dockerfile.gpu          # GPU-enabled Dockerfile
└── .env                    # Environment variables
```

## Development

### Backend Development

```bash
cd backend
pip install -r requirements.txt
python app.py
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### Running Tests

```bash
cd backend
pytest
```

## Monitoring and Observability

- **MLflow**: Access the MLflow UI at http://localhost:5001 to view experiment tracking, model metrics, and artifacts.
- **Weights & Biases**: Access your W&B dashboard at https://wandb.ai to view real-time training metrics, model performance, and system resource usage.

## Troubleshooting

### Docker Issues

If you encounter issues with Docker:

1. Make sure Docker and Docker Compose are installed and running
2. Try the simplified setup with `run-simple.bat` or `docker-compose -f docker-compose.simple.yml up -d`
3. Check Docker logs with `docker-compose logs`

### GPU Issues

If you're having trouble with GPU support:

1. Run `check-gpu.bat` to verify GPU compatibility
2. Ensure you have the latest NVIDIA drivers installed
3. Check that CUDA and cuDNN are properly installed
4. Try running with CPU only by setting `GPU_COUNT=0` in your `.env` file

## License

This project is licensed under the MIT License - see the LICENSE file for details.