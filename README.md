# Flappy Bird Reinforcement Learning

This project implements a reinforcement learning agent that learns to play Flappy Bird using Deep Q-Learning (DQN). The project includes real-time visualization of the training process, experiment tracking with MLflow, and performance monitoring.

## Features

- **Deep Q-Learning Agent**: Learns to play Flappy Bird through reinforcement learning
- **Real-time Training Visualization**: Watch the agent learn in real-time
- **Experiment Tracking**: Track experiments with MLflow and Weights & Biases
- **Performance Monitoring**: Monitor training progress and agent performance
- **Web Interface**: User-friendly web interface to control training and view results
- **Docker Support**: Development and production environments with CPU/GPU options

## Getting Started

### Prerequisites

- Docker and Docker Compose
- NVIDIA Docker (optional, for GPU acceleration)
- PowerShell (for Windows users)

### Running with Docker

#### Using Helper Scripts

We provide helper scripts for common Docker operations:

**Windows (PowerShell):**
```powershell
# Load the functions (run this first)
. .\docker-commands.ps1

# Development Environment
Start-DevApp                # Start with CPU
Start-DevAppGPU             # Start with GPU
Stop-App                    # Stop containers
Show-Logs [ServiceName]     # View logs (optional service name)

# Production Environment
Start-ProdApp               # Start production with CPU
Start-ProdAppGPU            # Start production with GPU
Stop-ProdApp                # Stop production containers
Show-ProdLogs [ServiceName] # View production logs

# Testing
Test-WebSocket              # Test WebSocket connection
```

#### Manual Docker Commands

##### Development Environment

1. Build and start the containers:
   ```bash
   # CPU Version
   docker-compose up --build
   
   # GPU Version
   docker-compose -f docker-compose.gpu.yml up --build
   ```

2. Open your browser and navigate to `http://localhost:3000`

##### Production Environment

1. Build and start the production containers:
   ```bash
   # CPU Version
   docker-compose -f docker-compose.prod.yml up -d --build
   
   # GPU Version
   docker-compose -f docker-compose.prod.gpu.yml up -d --build
   ```

2. Open your browser and navigate to `http://localhost` (port 80)

3. Stop the production containers:
   ```bash
   docker-compose -f docker-compose.prod.yml down
   ```

### Docker Configuration

The project includes several Docker configurations:

- **Development Environment**:
  - `docker-compose.yml`: Standard development setup with CPU
  - `docker-compose.gpu.yml`: Development setup with GPU support
  - `frontend/Dockerfile`: Frontend development container
  - `Dockerfile.backend`: Backend development container
  - `Dockerfile.backend.gpu`: Backend development container with GPU support

- **Production Environment**:
  - `docker-compose.prod.yml`: Production setup with CPU
  - `docker-compose.prod.gpu.yml`: Production setup with GPU support
  - `frontend/Dockerfile.prod`: Frontend production container with Nginx

### Project Structure

- `backend/`: Python backend code
  - `app.py`: Main Flask application
  - `training/`: Training-related code
    - `train.py`: Training manager and algorithms
  - `game/`: Game environment code
    - `flappy_bird.py`: Flappy Bird environment
  - `models/`: Saved model files

- `frontend/`: Next.js frontend code
  - `pages/`: Next.js pages
  - `components/`: React components
    - `TrainingVisualization.js`: Training visualization component
    - `GameComponent.js`: Game simulation component

## Testing the Visualization

To test the real-time training visualization:

1. Ensure the Docker containers are running
2. Navigate to `http://localhost:3000` (development) or `http://localhost` (production) in your browser
3. Click on the "Start Training" button
4. The training visualization should appear, showing:
   - The game environment with the bird and pipes
   - Training statistics (episode, score, etc.)
   - A chart showing training progress

You can also test the WebSocket connection directly:
```powershell
# Using the helper script
Test-WebSocket

# Or manually
Invoke-WebRequest -Uri "http://localhost:5000/api/status" -Method GET
```

## Troubleshooting

### WebSocket Connection Issues

If you're experiencing issues with the WebSocket connection:

1. Check that the backend container is running:
   ```powershell
   docker ps | Select-String backend
   ```

2. Check the backend logs:
   ```powershell
   # Development
   Show-Logs backend
   
   # Production
   Show-ProdLogs backend
   ```

3. Ensure your browser supports WebSockets
4. Check for any CORS issues in the browser console

### GPU Not Detected

If the GPU is not being detected:

1. Verify NVIDIA Docker is installed correctly:
   ```powershell
   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

2. Check the backend logs for GPU detection:
   ```powershell
   # Development
   Show-Logs backend | Select-String GPU
   
   # Production
   Show-ProdLogs backend | Select-String GPU
   ```

3. Make sure your GPU drivers are up to date

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym for the reinforcement learning framework
- Pygame for the game environment
- TensorFlow for the deep learning framework
- Next.js and React for the frontend framework 