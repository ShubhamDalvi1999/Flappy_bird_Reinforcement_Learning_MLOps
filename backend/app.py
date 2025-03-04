from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import numpy as np
import time
import threading
import mlflow
import wandb
from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent
from training.train import TrainingManager
import pandas as pd
from flask_socketio import SocketIO
from wandb_config import init_wandb

app = Flask(__name__)
CORS(app)
# Configure SocketIO with proper CORS settings
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

# Initialize MLflow - use environment variable or default to mlflow container
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)
print(f"MLflow tracking URI set to: {tracking_uri}")

# Add W&B status endpoint
@app.route('/api/wandb/status', methods=['GET'])
def get_wandb_status():
    """Check Weights & Biases connection status"""
    try:
        api_key = os.getenv('WANDB_API_KEY')
        entity = os.getenv('WANDB_ENTITY')
        project = os.getenv('WANDB_PROJECT', 'flappy-bird-rl')
        
        if not api_key:
            return jsonify({
                'status': 'error',
                'message': 'WANDB_API_KEY environment variable is not set'
            })
            
        # Try to initialize a test run
        test_run = init_wandb({
            'name': 'connection-test',
            'entity': entity,
            'project': project
        })
        
        # Get recent runs
        api = wandb.Api()
        runs = list(api.runs(f"{entity}/{project}" if entity else project))
        recent_runs = []
        
        for run in runs[:10]:  # Get last 10 runs
            recent_runs.append({
                'id': run.id,
                'name': run.name,
                'state': run.state,
                'created_at': run.created_at,
                'config': run.config,
                'summary': {k: v for k, v in run.summary.items() if not k.startswith('_')}
            })
        
        # Clean up test run
        test_run.finish()
        
        return jsonify({
            'status': 'success',
            'message': 'Successfully connected to W&B',
            'entity': entity,
            'project': project,
            'recent_runs': recent_runs
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error connecting to W&B: {str(e)}'
        })

# Add GPU check endpoint
@app.route('/api/gpu_check', methods=['GET'])
def check_gpu():
    """Check if TensorFlow can access GPUs"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info = []
        
        if gpus:
            # Get detailed information about available GPUs
            for gpu in gpus:
                try:
                    gpu_attrs = tf.config.experimental.get_device_details(gpu)
                    gpu_info.append({
                        'name': gpu.name,
                        'type': gpu.device_type,
                        'details': gpu_attrs
                    })
                except Exception as e:
                    gpu_info.append({
                        'name': gpu.name,
                        'type': gpu.device_type,
                        'details': str(e)
                    })
        
        return jsonify({
            'status': 'success',
            'gpu_available': len(gpus) > 0,
            'gpu_count': len(gpus),
            'gpu_info': gpu_info,
            'tf_version': tf.__version__
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'gpu_available': False,
            'error': str(e)
        })

# Global variable for training manager
training_manager = None

# Initialize training manager
def init_training_manager():
    global training_manager
    if training_manager is None:
        try:
            print("Initializing training manager...")
            training_manager = TrainingManager(socketio=socketio)
            print("Training manager initialized successfully")
        except Exception as e:
            print(f"Error initializing training manager: {str(e)}")
            return False
    return True

# Game session
game_session = {
    "env": None,
    "agent": None,
    "active": False,
    "state": None
}

def initialize_mlflow():
    """Initialize MLflow tracking"""
    try:
        # Set tracking URI
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
        
        # Create experiment if it doesn't exist
        experiment_name = "flappy-bird-rl"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
            print(f"Created MLflow experiment: {experiment_name}")
        else:
            print(f"Using existing MLflow experiment: {experiment_name}")
        
        return True
    except Exception as e:
        print(f"Error initializing MLflow: {str(e)}")
        return False

def initialize_app():
    """Initialize the application components"""
    # Create necessary directories
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Initialize MLflow
    mlflow_initialized = initialize_mlflow()
    if not mlflow_initialized:
        print("Warning: MLflow initialization failed")
    
    # Initialize training manager
    training_manager_initialized = init_training_manager()
    if not training_manager_initialized:
        print("Warning: Training manager initialization failed")
    
    # Create a default model if none exists
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.pth')]
    if not model_files:
        print("Creating a default model file...")
        try:
            env = FlappyBirdEnv()
            agent = DQNAgent(state_size=4, action_size=2)
            default_model_path = os.path.join(models_dir, 'default_model.h5')
            agent.save(default_model_path)
            print(f"Default model created at {default_model_path}")
            
            # Log default model to MLflow if initialized
            if mlflow_initialized:
                with mlflow.start_run(description="Default model creation") as run:
                    mlflow.log_artifact(default_model_path)
                    mlflow.log_param("model_type", "default")
        except Exception as e:
            print(f"Error creating default model: {e}")

# Initialize the app
initialize_app()

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        return jsonify({
            'status': 'success',
            'models': []
        })
    
    models = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.h5') or filename.endswith('.pth'):
            model_path = os.path.join(models_dir, filename)
            created_time = os.path.getmtime(model_path)
            
            # Extract model name and timestamp
            name = filename.replace('.h5', '').replace('.pth', '')
            
            models.append({
                'id': name,
                'name': name,
                'created_at': created_time * 1000  # Convert to milliseconds for JS
            })
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x['created_at'], reverse=True)
    
    return jsonify({
        'status': 'success',
        'models': models
    })

@app.route('/api/train', methods=['POST'])
def start_training():
    """Start training with optional base model and configuration"""
    try:
        data = request.get_json()
        
        # Episode validation and configuration
        episodes = data.get('episodes', 1000)
        if not isinstance(episodes, int) or episodes < 1:
            return jsonify({
                "status": "error",
                "message": "Episodes must be a positive integer"
            })
        
        # Cap maximum episodes for safety
        max_episodes = int(os.getenv('MAX_TRAINING_EPISODES', 10000))
        if episodes > max_episodes:
            return jsonify({
                "status": "warning",
                "message": f"Episodes capped at {max_episodes}",
                "original_episodes": episodes,
                "adjusted_episodes": max_episodes
            })
            episodes = max_episodes
        
        # Get optional configuration
        model_id = data.get('model_id')
        batch_size = data.get('batch_size', 32)
        run_name = data.get('run_name')
        use_wandb = data.get('use_wandb', True)
        use_mlflow = data.get('use_mlflow', True)
        
        # Validate model_id if provided
        if model_id:
            models_dir = os.path.join(os.path.dirname(__file__), 'models')
            model_path = os.path.join(models_dir, f'{model_id}.h5')
            if not os.path.exists(model_path):
                return jsonify({
                    "status": "error",
                    "message": f"Model {model_id} not found"
                })
        
        # Estimate training time (rough estimate)
        estimated_time = episodes * 0.5  # 0.5 seconds per episode estimate
        
        # Start training in a background thread
        def train_async():
            try:
                success, message = training_manager.train(
                    episodes=episodes,
                    batch_size=batch_size,
                    model_id=model_id,
                    run_name=run_name,
                    use_wandb=use_wandb,
                    use_mlflow=use_mlflow
                )
                socketio.emit('training_complete', {
                    'status': 'success' if success else 'error',
                    'message': message
                })
            except Exception as e:
                socketio.emit('training_error', {
                    'status': 'error',
                    'message': str(e)
                })
        
        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Training started",
            "base_model": model_id if model_id else "new",
            "config": {
                "episodes": episodes,
                "batch_size": batch_size,
                "estimated_time_seconds": estimated_time,
                "use_wandb": use_wandb,
                "use_mlflow": use_mlflow
            }
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/stop', methods=['POST'])
def stop_training():
    training_manager.stop_training()
    return jsonify({"status": "success", "message": "Training stopped"})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = training_manager.get_training_stats()
    if stats is None:
        return jsonify({"status": "error", "message": "No training data available"})
    return jsonify({"status": "success", "data": stats})

@app.route('/api/analysis', methods=['GET'])
def get_analysis():
    analysis = training_manager.analyze_training_data()
    if analysis is None:
        return jsonify({"status": "error", "message": "No training data available for analysis"})
    return jsonify({"status": "success", "data": analysis})

@app.route('/api/wandb/status', methods=['GET'])
def get_wandb_status():
    try:
        is_logged_in = wandb.api.api_key is not None
        return jsonify({
            "status": "success",
            "logged_in": is_logged_in
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/mlflow/experiments', methods=['GET'])
def get_mlflow_experiments():
    """Get MLflow experiments"""
    try:
        experiments = mlflow.search_experiments()
        experiment_data = []
        
        for exp in experiments:
            experiment_data.append({
                'experiment_id': exp.experiment_id,
                'name': exp.name,
                'artifact_location': exp.artifact_location,
                'lifecycle_stage': exp.lifecycle_stage,
                'creation_time': exp.creation_time
            })
        
        return jsonify({
            'status': 'success',
            'experiments': experiment_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/mlflow/runs', methods=['GET'])
def get_mlflow_runs():
    """Get runs for a specific experiment"""
    experiment_id = request.args.get('experiment_id')
    if not experiment_id:
        return jsonify({
            'status': 'error',
            'message': 'experiment_id is required'
        })
    
    try:
        # First check if the experiment exists
        try:
            experiment = mlflow.get_experiment(experiment_id)
            if not experiment:
                return jsonify({
                    'status': 'error',
                    'message': f'Experiment with ID {experiment_id} not found'
                })
        except Exception as exp_error:
            print(f"Error getting experiment: {exp_error}")
            # Try to create a default experiment and use it
            try:
                print("Creating a default experiment as fallback...")
                default_exp_id = mlflow.create_experiment("Default Experiment")
                experiment_id = default_exp_id
            except:
                # Return an empty list if we can't create an experiment
                return jsonify({
                    'status': 'success',
                    'runs': [],
                    'warning': 'Could not access MLflow experiments. Using empty dataset.'
                })
        
        # Now get the runs
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        run_data = []
        
        # If there are no runs, create a sample run for demonstration
        if runs.empty:
            print("No runs found, creating a sample run")
            try:
                with mlflow.start_run(experiment_id=experiment_id) as run:
                    mlflow.log_param("learning_rate", 0.001)
                    mlflow.log_param("gamma", 0.99)
                    mlflow.log_metric("reward", 10.0)
                    mlflow.log_metric("loss", 0.5)
                # Fetch the newly created run
                runs = mlflow.search_runs(experiment_ids=[experiment_id])
            except Exception as run_error:
                print(f"Error creating sample run: {run_error}")
        
        for _, run in runs.iterrows():
            metrics = {}
            for key in run.keys():
                if key.startswith('metrics.'):
                    metric_name = key.replace('metrics.', '')
                    metrics[metric_name] = run[key]
            
            # Handle NaN values to avoid JSON serialization errors
            start_time = None
            if 'start_time' in run and not pd.isna(run['start_time']):
                start_time = run['start_time'] * 1000
            
            end_time = None
            if 'end_time' in run and not pd.isna(run['end_time']):
                end_time = run['end_time'] * 1000
            
            run_data.append({
                'run_id': run['run_id'],
                'experiment_id': run['experiment_id'],
                'status': run['status'],
                'start_time': start_time,
                'end_time': end_time,
                'metrics': metrics,
                'tags': {k.replace('tags.', ''): v for k, v in run.items() if k.startswith('tags.') and not pd.isna(v)}
            })
        
        return jsonify({
            'status': 'success',
            'runs': run_data
        })
    except Exception as e:
        print(f"Error in MLflow runs endpoint: {e}")
        # Return an empty dataset with an error message
        return jsonify({
            'status': 'success',
            'runs': [],
            'warning': f'Error retrieving MLflow data: {str(e)}'
        })

@app.route('/api/models/<model_id>/metrics', methods=['GET'])
def get_model_metrics(model_id):
    """Get metrics for a specific model from MLflow"""
    try:
        # Search for runs that used this model
        runs = mlflow.search_runs(
            filter_string=f"tags.model_id = '{model_id}'"
        )
        
        if runs.empty:
            return jsonify({
                'status': 'error',
                'message': f'No metrics found for model {model_id}'
            })
        
        # Get the latest run for this model
        latest_run = runs.iloc[0]
        
        # Extract metrics from the run
        metrics = {
            'max_score': float(latest_run.get('metrics.max_score', 0)),
            'avg_score': float(latest_run.get('metrics.average_score', 0)),
            'episodes_trained': int(latest_run.get('metrics.episodes_trained', 0)),
            'learning_rate': float(latest_run.get('params.learning_rate', 0.001)),
            'discount_factor': float(latest_run.get('params.gamma', 0.99)),
            'training_time': float(latest_run.get('metrics.training_time_minutes', 0)),
            'architecture': latest_run.get('params.architecture', 'DQN'),
            'epsilon_final': float(latest_run.get('metrics.final_epsilon', 0)),
            'total_steps': int(latest_run.get('metrics.total_steps', 0)),
            'best_episode_reward': float(latest_run.get('metrics.best_episode_reward', 0))
        }
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        print(f"Error fetching model metrics: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to fetch metrics: {str(e)}'
        })

@app.route('/api/models/<model_id>/history', methods=['GET'])
def get_model_history(model_id):
    """Get training history for a specific model"""
    try:
        # This would typically come from a database or MLflow
        # For now, generating example data
        episodes = list(range(0, 1000, 10))
        scores = [max(0, min(100, 50 + i/10 + np.random.normal(0, 15))) for i in range(len(episodes))]
        
        return jsonify({
            'status': 'success',
            'history': {
                'episodes': episodes,
                'scores': scores
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/game/start', methods=['POST'])
def start_game():
    """Start a game session with the selected model"""
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({
            'status': 'error',
            'message': 'model_id is required'
        })
    
    try:
        # Create environment and agent
        game_session['env'] = FlappyBirdEnv()
        game_session['agent'] = DQNAgent(
            state_size=game_session['env'].state_size,
            action_size=game_session['env'].action_size
        )
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_id}.pth')
        game_session['agent'].load(model_path)
        
        # Initialize game state
        state = game_session['env'].reset()
        game_session['state'] = state
        game_session['active'] = True
        
        # Prepare initial state for frontend
        initial_state = {
            'bird_x': game_session['env'].bird_x,
            'bird_y': game_session['env'].bird_y,
            'bird_velocity': game_session['env'].bird_velocity,
            'pipes': game_session['env'].pipes,
            'pipe_gap': game_session['env'].pipe_gap,
            'score': game_session['env'].score,
            'done': False
        }
        
        return jsonify({
            'status': 'success',
            'message': 'Game started',
            'initial_state': initial_state
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to start game: {str(e)}'
        })

@app.route('/api/game/step', methods=['GET'])
def game_step():
    """Get the next frame of the game"""
    if not game_session['active']:
        return jsonify({
            'status': 'error',
            'message': 'No active game session'
        })
    
    try:
        # Get agent's action
        state = game_session['state']
        action = game_session['agent'].act(state, 0)  # Epsilon 0 for inference
        
        # Apply action and get new state
        next_state, reward, done, _ = game_session['env'].step(action)
        game_session['state'] = next_state
        
        # Prepare state for frontend
        state_data = {
            'bird_x': game_session['env'].bird_x,
            'bird_y': game_session['env'].bird_y,
            'bird_velocity': game_session['env'].bird_velocity,
            'pipes': game_session['env'].pipes,
            'pipe_gap': game_session['env'].pipe_gap,
            'score': game_session['env'].score,
            'done': done
        }
        
        # If game over, reset the session
        if done:
            game_session['active'] = False
        
        return jsonify({
            'status': 'success',
            'state': state_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during game step: {str(e)}'
        })

@app.route('/api/game/stop', methods=['POST'])
def stop_game():
    """Stop the current game session"""
    game_session['active'] = False
    if game_session['env']:
        game_session['env'].close()
    
    return jsonify({
        'status': 'success',
        'message': 'Game stopped'
    })

@app.route('/api/analytics/summary', methods=['GET'])
def get_analytics_summary():
    """Get analytics summary using pandas"""
    try:
        # Use pandas instead of Spark for analytics
        # This would typically analyze real data, but for now just a placeholder
        summary = {
            'total_games': 1250,
            'average_score': 42.5,
            'max_score': 156,
            'training_time_hrs': 8.5
        }
        
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Docker"""
    try:
        # Check MLflow connection
        mlflow_ok = False
        mlflow_error = None
        try:
            # Simple check that doesn't require database access
            assert mlflow.get_tracking_uri() == tracking_uri
            # Try a lightweight operation
            mlflow.search_experiments(max_results=1)
            mlflow_ok = True
        except Exception as e:
            mlflow_error = str(e)
            print(f"MLflow health check failed: {mlflow_error}")

        # Check if training manager is initialized or can be initialized
        training_ok = training_manager is not None
        if not training_ok:
            training_ok = init_training_manager()

        # Overall health status
        is_healthy = mlflow_ok and training_ok

        response = {
            'status': 'healthy' if is_healthy else 'unhealthy',
            'checks': {
                'mlflow': {
                    'status': 'ok' if mlflow_ok else 'error',
                    'tracking_uri': tracking_uri
                },
                'training_manager': {
                    'status': 'ok' if training_ok else 'error',
                    'initialized': training_manager is not None
                }
            }
        }

        if not is_healthy:
            if not mlflow_ok and mlflow_error:
                response['checks']['mlflow']['error'] = mlflow_error

        return jsonify(response), 200 if is_healthy else 503
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

# MLflow UI redirect
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Serve MLflow UI at root"""
    # This assumes MLflow UI is available at port 5000
    # In a real deployment, you might use nginx or a similar server for this
    return f"MLflow UI is available at <a href='/mlflow/'>/mlflow/</a>"

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    if not init_training_manager():
        return jsonify({
            "status": "error",
            "message": "Training manager not initialized"
        })
    
    try:
        is_training = training_manager.is_training
        message = "Training in progress" if is_training else "Not training"
        
        # Get additional stats if available
        stats = None
        if hasattr(training_manager, 'episode_rewards') and training_manager.episode_rewards:
            stats = {
                'episodes_completed': len(training_manager.episode_rewards),
                'latest_reward': training_manager.episode_rewards[-1] if training_manager.episode_rewards else 0,
                'latest_score': training_manager.episode_scores[-1] if training_manager.episode_scores else 0,
                'best_score': training_manager.best_score
            }
        
        return jsonify({
            "status": "success",
            "is_training": is_training,
            "message": message,
            "stats": stats
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/training/start', methods=['POST'])
def start_training_with_viz():
    """Start training with visualization enabled"""
    if not init_training_manager():
        return jsonify({
            "status": "error",
            "message": "Training manager not initialized"
        })
    
    try:
        data = request.get_json() or {}
        
        # Get training parameters
        episodes = data.get('episodes', 1000)
        batch_size = data.get('batch_size', 32)
        model_id = data.get('model_id')
        save_interval = data.get('save_interval', 100)
        
        # Start training in a background thread
        def train_async():
            try:
                success, message = training_manager.train(
                    episodes=episodes,
                    batch_size=batch_size,
                    model_id=model_id,
                    run_name=f"training_{int(time.time())}",
                    use_wandb=True,
                    use_mlflow=True
                )
                socketio.emit('training_complete', {
                    'status': 'success' if success else 'error',
                    'message': message
                })
            except Exception as e:
                socketio.emit('training_error', {
                    'status': 'error',
                    'message': str(e)
                })
        
        thread = threading.Thread(target=train_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Training started with visualization",
            "config": {
                "episodes": episodes,
                "batch_size": batch_size,
                "model_id": model_id,
                "save_interval": save_interval
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/test/socket', methods=['GET'])
def test_socket_connection():
    """Test WebSocket connection by emitting a test frame"""
    try:
        # Create a simple test image
        import pygame
        import io
        import base64
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        # Create a test surface
        width, height = 400, 600
        test_surface = pygame.Surface((width, height))
        test_surface.fill((135, 206, 235))  # Sky blue background
        
        # Draw some shapes
        pygame.draw.circle(test_surface, (255, 255, 0), (width//2, height//2), 50)  # Yellow circle
        font = pygame.font.SysFont("Arial", 30)
        text = font.render("WebSocket Test", True, (0, 0, 0))
        test_surface.blit(text, (width//2 - text.get_width()//2, height//2 - text.get_height()//2))
        
        # Convert to base64
        buffer = io.BytesIO()
        pygame.image.save(test_surface, buffer, "PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Emit test frame
        socketio.emit('game_frame', {
            'frame': img_base64,
            'score': 0,
            'episode': 0,
            'epsilon': 1.0
        })
        
        # Emit test training progress
        socketio.emit('training_progress', {
            'episode': 1,
            'total_episodes': 100,
            'score': 0,
            'reward': 0,
            'best_score': 0,
            'epsilon': 1.0
        })
        
        return jsonify({
            "status": "success",
            "message": "Test frame emitted via WebSocket"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/api/training/stop', methods=['POST'])
def stop_training_viz():
    """Stop the current training process"""
    if not init_training_manager():
        return jsonify({
            "status": "error",
            "message": "Training manager not initialized"
        })
    
    try:
        # Set is_training to False to stop the training loop
        if hasattr(training_manager, 'is_training'):
            training_manager.is_training = False
            
            # Emit training stopped event
            socketio.emit('training_complete', {
                'status': 'success',
                'message': 'Training stopped by user'
            })
            
            return jsonify({
                "status": "success",
                "message": "Training stopped"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Training manager does not have is_training attribute"
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    
    # Start the app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 