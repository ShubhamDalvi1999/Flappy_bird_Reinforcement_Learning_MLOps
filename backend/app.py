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

app = Flask(__name__)
CORS(app)

# Initialize MLflow
mlflow.set_tracking_uri("http://localhost:5000")
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

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
        if filename.endswith('.pth'):
            model_path = os.path.join(models_dir, filename)
            created_time = os.path.getmtime(model_path)
            
            # Extract model name and timestamp
            name = filename.replace('.pth', '')
            
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
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        run_data = []
        
        for _, run in runs.iterrows():
            metrics = {}
            for key in run.keys():
                if key.startswith('metrics.'):
                    metric_name = key.replace('metrics.', '')
                    metrics[metric_name] = run[key]
            
            run_data.append({
                'run_id': run['run_id'],
                'experiment_id': run['experiment_id'],
                'status': run['status'],
                'start_time': run['start_time'] * 1000 if not np.isnan(run['start_time']) else None,
                'end_time': run['end_time'] * 1000 if not np.isnan(run['end_time']) else None,
                'metrics': metrics,
                'tags': {k.replace('tags.', ''): v for k, v in run.items() if k.startswith('tags.') and not pd.isna(v)}
            })
        
        return jsonify({
            'status': 'success',
            'runs': run_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
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