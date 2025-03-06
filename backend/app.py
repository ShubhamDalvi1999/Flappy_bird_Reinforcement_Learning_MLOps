from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import numpy as np
import time
import threading
from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent
from training.train import TrainingManager
import pandas as pd
from flask_socketio import SocketIO

app = Flask(__name__)
CORS(app)
# Configure SocketIO with proper CORS settings
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', logger=True, engineio_logger=True)

# Global variables
training_manager = None
game_env = None
game_agent = None
is_game_running = False
game_thread = None

# Status endpoint
@app.route('/api/status', methods=['GET'])
def get_status():
    """Check API status"""
    return jsonify({
        'status': 'success',
        'message': 'API is running',
        'version': '1.0.0'
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
                        'error': str(e)
                    })
            
            return jsonify({
                'status': 'success',
                'gpu_available': True,
                'gpu_count': len(gpus),
                'gpu_info': gpu_info
            })
        else:
            return jsonify({
                'status': 'success',
                'gpu_available': False,
                'message': 'No GPUs detected. Running in CPU mode.'
            })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error checking GPU: {str(e)}'
        })

# Training endpoints
@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training the agent"""
    global training_manager
    
    try:
        data = request.json or {}
        episodes = data.get('episodes', 1000)
        batch_size = data.get('batch_size', 32)
        model_id = data.get('model_id', None)
        
        if training_manager is None:
            training_manager = TrainingManager(socketio=socketio)
        
        if training_manager.is_training:
            return jsonify({
                'status': 'error',
                'message': 'Training is already in progress'
            })
        
        # Start training in a separate thread
        def train_thread():
            training_manager.train(
                episodes=episodes,
                batch_size=batch_size,
                model_id=model_id,
                use_wandb=False,
                use_mlflow=False
            )
        
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Training started with {episodes} episodes'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error starting training: {str(e)}'
        })

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop the training process"""
    global training_manager
    
    try:
        if training_manager is None or not training_manager.is_training:
            return jsonify({
                'status': 'error',
                'message': 'No training in progress'
            })
        
        training_manager.stop_training()
        
        return jsonify({
            'status': 'success',
            'message': 'Training stopped'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error stopping training: {str(e)}'
        })

@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get the current training status"""
    global training_manager
    
    try:
        if training_manager is None:
            return jsonify({
                'status': 'success',
                'is_training': False,
                'message': 'No training manager initialized'
            })
        
        return jsonify({
            'status': 'success',
            'is_training': training_manager.is_training,
            'message': 'Training in progress' if training_manager.is_training else 'No training in progress',
            'metrics': {
                'episode': training_manager.current_episode if hasattr(training_manager, 'current_episode') else 0,
                'best_score': training_manager.best_score if hasattr(training_manager, 'best_score') else 0
            }
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting training status: {str(e)}'
        })

# Model endpoints
@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    try:
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        models = []
        for file in os.listdir(models_dir):
            if file.endswith('.h5'):
                model_id = file.replace('.h5', '')
                model_path = os.path.join(models_dir, file)
                models.append({
                    'id': model_id,
                    'name': model_id,
                    'path': model_path,
                    'size': os.path.getsize(model_path),
                    'created': os.path.getctime(model_path)
                })
        
        return jsonify({
            'status': 'success',
            'models': models
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error getting models: {str(e)}'
        })

# Game endpoints
@app.route('/api/game/start', methods=['POST'])
def start_game():
    """Start a game with the selected model"""
    global game_env, game_agent, is_game_running, game_thread
    
    try:
        data = request.json or {}
        model_id = data.get('model_id')
        
        if is_game_running:
            return jsonify({
                'status': 'error',
                'message': 'Game is already running'
            })
        
        if not model_id:
            return jsonify({
                'status': 'error',
                'message': 'No model selected'
            })
        
        # Initialize environment and agent
        game_env = FlappyBirdEnv()
        state_size = game_env.observation_space.shape[0]
        action_size = game_env.action_space.n
        
        game_agent = DQNAgent(state_size, action_size)
        
        # Load model
        model_path = os.path.join(os.path.dirname(__file__), 'models', f'{model_id}.h5')
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found'
            })
        
        game_agent.load(model_id)
        
        # Start game in a separate thread
        def game_loop():
            global is_game_running
            
            state, _ = game_env.reset()
            done = False
            score = 0
            
            while is_game_running and not done:
                # Get action from agent
                action = game_agent.act(state, explore=False)
                
                # Take action
                next_state, reward, done, _, info = game_env.step(action)
                
                # Update state
                state = next_state
                
                # Update score
                score = info.get('score', score)
                
                # Render game
                frame = game_env.render()
                
                # Emit frame to client
                socketio.emit('game_frame', {
                    'frame': base64.b64encode(frame).decode('utf-8'),
                    'score': score,
                    'action': int(action)
                })
                
                # Sleep to control frame rate
                time.sleep(0.05)
            
            is_game_running = False
            socketio.emit('game_over', {
                'score': score
            })
        
        # Start game thread
        is_game_running = True
        game_thread = threading.Thread(target=game_loop)
        game_thread.daemon = True
        game_thread.start()
        
        # Get initial state
        initial_state, _ = game_env.reset()
        initial_frame = game_env.render()
        
        return jsonify({
            'status': 'success',
            'message': 'Game started',
            'initial_state': initial_state.tolist(),
            'initial_frame': base64.b64encode(initial_frame).decode('utf-8')
        })
    except Exception as e:
        is_game_running = False
        return jsonify({
            'status': 'error',
            'message': f'Error starting game: {str(e)}'
        })

@app.route('/api/game/stop', methods=['POST'])
def stop_game():
    """Stop the current game"""
    global is_game_running
    
    try:
        if not is_game_running:
            return jsonify({
                'status': 'error',
                'message': 'No game in progress'
            })
        
        is_game_running = False
        
        return jsonify({
            'status': 'success',
            'message': 'Game stopped'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error stopping game: {str(e)}'
        })

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
    
    # Start the server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 