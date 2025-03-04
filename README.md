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
- **Spark Cluster**: Handles data processing and analytics
- **Docker**: Ensures consistent deployment across environments

## Getting Started

### Prerequisites

- Docker and Docker Compose
- [Optional] Weights & Biases account for enhanced monitoring

### Installation

#### Linux/macOS

1. Clone the repository:

```bash
git clone https://github.com/yourusername/flappy-bird-rl.git
cd flappy-bird-rl
```

2. [Optional] Set up your Weights & Bibes API key:

```bash
export WANDB_API_KEY=your_api_key_here
```

3. Start the application using Docker Compose:

```bash
./run.sh start
```

#### Windows

For Windows users, we provide special instructions and scripts to ensure smooth operation:

1. Clone the repository:

```cmd
git clone https://github.com/yourusername/flappy-bird-rl.git
cd flappy-bird-rl
```

2. [Optional] Set up your Weights & Bibes API key:

```cmd
set WANDB_API_KEY=your_api_key_here
```

3. Start the application using the provided batch script:

```cmd
run.bat start
```

For more detailed Windows-specific instructions, including troubleshooting and performance optimization, please see [WINDOWS.md](WINDOWS.md).

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000/api
- MLflow UI: http://localhost:5001
- Spark Master UI: http://localhost:8080

### Training a Model

1. Open the web interface at http://localhost:3000
2. Navigate to the "Game" tab
3. Click "Start Training" to begin training a new model
4. Monitor progress through the UI or MLflow/W&B dashboards

### Playing with a Trained Model

1. Select a trained model from the dropdown
2. Click "Start Game" to see the AI play Flappy Bird
3. Watch the game and observe metrics in real-time

## Project Structure

```
flappy-bird-rl/
├── backend/                # Python backend code
│   ├── agent/              # DQN agent implementation
│   ├── game/               # Flappy Bird environment
│   ├── training/           # Training logic
│   ├── tests/              # Test files
│   ├── app.py              # Flask application
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── components/         # React components
│   ├── pages/              # Next.js pages
│   ├── public/             # Static assets
│   └── package.json        # Node.js dependencies
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Backend Dockerfile
└── README.md               # This file
```

## Monitoring and Observability

### MLflow

MLflow is used for experiment tracking, model versioning, and artifact storage. Access the MLflow UI at http://localhost:5001 to:

- Compare experiment runs
- View model parameters and metrics
- Download trained models
- Register models in the model registry

### Weights & Biases

Weights & Biases is used for enhanced visualization and monitoring of training metrics. It provides:

- Real-time training metrics visualization
- Custom charts and histograms for performance analysis
- Automatic tracking of GPU utilization and system metrics
- Interactive dashboards for experiment comparison
- Training alerts for significant events
- Model architecture visualization

Access your W&B dashboard at https://wandb.ai/ after logging in with your API key.

New features include:
- **Action Distribution Visualization**: See how often the agent chooses each action
- **Performance Summary Tables**: Track detailed metrics in table format
- **Histograms of Scores**: Monitor the distribution of scores over time
- **Enhanced Model Summaries**: Get detailed insights into the agent's performance
- **Integrated GPU Monitoring**: Track GPU utilization directly in the dashboard

The application provides a built-in W&B dashboard for viewing these metrics directly within the UI without leaving the application.

## Testing

Run tests using pytest:

```bash
docker-compose exec backend pytest
```

For test coverage:

```bash
docker-compose exec backend pytest --cov=.
```

## Customization

### Hyperparameters

You can modify hyperparameters through the frontend interface or by editing `backend/agent/dqn_agent.py`.

### Model Architecture

The neural network architecture can be customized in `backend/agent/dqn_agent.py`.

### Environment

Flappy Bird game parameters can be adjusted in `backend/game/flappy_bird.py`.

## Technical Details

### Deep Q-Learning Algorithm

The application uses Deep Q-Learning with experience replay and target networks to train the agent:

1. The agent observes the game state (bird position, velocity, pipe positions)
2. Based on the state, it chooses an action (flap or don't flap)
3. The action is executed, resulting in a new state and reward
4. The experience (state, action, reward, next state) is stored in replay memory
5. Periodically, the agent learns by sampling from replay memory
6. The process continues until the agent learns optimal behavior

### PySpark Analytics

PySpark is used for:

- Analyzing training data
- Computing correlations between metrics
- Processing large datasets for model evaluation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for research on Deep Q-Learning
- The PyTorch team for the deep learning framework
- MLflow, W&B, and Apache Spark communities for their excellent tools