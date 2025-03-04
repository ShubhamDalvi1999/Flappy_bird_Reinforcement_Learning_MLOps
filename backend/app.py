from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import numpy as np
import time
import threading
import mlflow
import wandb
from pyspark.sql import SparkSession
from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent
from training.train import TrainingManager
import pandas as pd

app = Flask(__name__)
CORS(app)

# Initialize MLflow - use environment variable or default to mlflow container
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(tracking_uri)
print(f"MLflow tracking URI set to: {tracking_uri}")

# Add W&B status endpoint
@app.route('/api/wandb/status', methods=['GET'])
def check_wandb_status():
    """Check Weights & Biases connection status and recent runs"""
    try:
        # Check if WANDB_API_KEY is set
        api_key = os.environ.get('WANDB_API_KEY')
        if not api_key:
            return jsonify({
                'status': 'success',
                'wandb_status': 'not_configured',
                'message': 'WANDB_API_KEY environment variable is not set'
            })
        
        # Try to initialize wandb API
        api = wandb.Api()
        
        # Get recent runs (up to 10)
        entity = os.environ.get('WANDB_ENTITY') 
        project = os.environ.get('WANDB_PROJECT', 'flappy-bird-rl')
        
        try:
            recent_runs = []
            
            if entity:
                runs = api.runs(f"{entity}/{project}", per_page=10)
            else:
                # Try to get the default entity from the API
                runs = api.runs(f"{project}", per_page=10)
            
            for run in runs:
                run_data = {
                    'id': run.id,
                    'name': run.name,
                    'state': run.state,
                    'created_at': run.created_at,
                    'summary': {k: v for k, v in run.summary.items()}
                }
                recent_runs.append(run_data)
            
            return jsonify({
                'status': 'success',
                'wandb_status': 'connected',
                'message': 'Successfully connected to W&B',
                'recent_runs': recent_runs
            })
            
        except Exception as e:
            # Could connect to W&B but couldn't fetch runs
            return jsonify({
                'status': 'success',
                'wandb_status': 'connected',
                'message': f'Connected to W&B but could not fetch runs: {str(e)}',
                'recent_runs': []
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'wandb_status': 'error',
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

# Initialize training manager
training_manager = TrainingManager()

# Game session
game_session = {
    "env": None,
    "agent": None,
    "active": False,
    "state": None
}

# Spark session for data analysis
spark = None

def init_spark():
    global spark
    if spark is None:
        spark = SparkSession.builder \
            .appName("FlappyBirdAnalytics") \
            .config("spark.executor.memory", "2g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
    return spark

def initialize_app():
    """Initialize the application by creating necessary directories and default models."""
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a default model file if none exists
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.pth')]
    if not model_files:
        print("Creating a default model file...")
        # Create a simple DQN agent and save it
        env = FlappyBirdEnv()
        agent = DQNAgent(state_size=4, action_size=2)
        default_model_path = 'default_model'
        agent.save(default_model_path)
        print(f"Default model created at {default_model_path}.h5")
    
    # Create a default MLflow experiment if none exists
    try:
        experiments = mlflow.search_experiments()
        if not experiments:
            print("Creating a default MLflow experiment...")
            mlflow.create_experiment("Default Experiment")
            print("Default MLflow experiment created")
    except Exception as e:
        print(f"Error creating MLflow experiment: {e}")

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

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training process"""
    data = request.json
    episodes = data.get('episodes', 1000)
    save_interval = data.get('save_interval', 100)
    
    # Start training in a separate thread
    def training_task():
        training_manager.train(
            episodes=episodes,
            save_interval=save_interval,
            use_mlflow=True,
            use_wandb=True
        )
    
    thread = threading.Thread(target=training_task)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'status': 'success',
        'message': f'Training started with {episodes} episodes'
    })

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the training process"""
    training_manager.stop_training()
    return jsonify({
        'status': 'success',
        'message': 'Training stopped'
    })

@app.route('/api/training/status', methods=['GET'])
def training_status():
    """Get current training status"""
    status = training_manager.get_status()
    return jsonify({
        'status': 'success',
        'is_training': status['is_training'],
        'message': status['message'],
        'progress': status.get('progress', {})
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
    """Get metrics for a specific model"""
    try:
        # Here we would retrieve metrics from MLflow or a database
        # For simplicity, generating some example metrics
        metrics = {
            'max_score': 125,
            'avg_score': 56.8,
            'episodes_trained': 1000,
            'learning_rate': 0.001,
            'discount_factor': 0.99,
            'training_time': 45,  # minutes
            'architecture': 'DQN (3 layers)'
        }
        
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
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
    """Get analytics summary using Spark"""
    try:
        spark = init_spark()
        
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

# MLflow UI redirect
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    """Serve MLflow UI at root"""
    # This assumes MLflow UI is available at port 5000
    # In a real deployment, you might use nginx or a similar server for this
    return f"MLflow UI is available at <a href='/mlflow/'>/mlflow/</a>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 