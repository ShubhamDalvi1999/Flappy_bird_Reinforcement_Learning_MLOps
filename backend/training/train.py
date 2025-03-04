import os
import time
import json
import numpy as np
import torch
import mlflow
import wandb
from datetime import datetime
from flask_socketio import SocketIO
import base64

from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent
from wandb_config import init_wandb, log_episode_metrics, log_model_summary

class TrainingManager:
    def __init__(self, socketio=None):
        self.env = None
        self.agent = None
        self.is_training = False
        self.socketio = socketio
        self.episode_rewards = []
        self.episode_scores = []
        self.best_score = 0
        self.wandb_run = None
        self.current_model_id = None

    def load_model(self, model_id=None):
        """Load a specific model or create a new one"""
        try:
            self.env = FlappyBirdEnv()
            state_size = self.env.observation_space.shape[0]
            action_size = self.env.action_space.n
            
            self.agent = DQNAgent(state_size, action_size)
            
            if model_id:
                model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{model_id}.h5')
                if os.path.exists(model_path):
                    print(f"Loading model from {model_path}")
                    self.agent.load(model_path)
                    self.current_model_id = model_id
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def initialize_training(self, model_id=None):
        """Initialize the environment and agent for training"""
        if not self.load_model(model_id):
            return False
            
        # Initialize MLflow run
        mlflow.start_run()
        mlflow.log_param("state_size", self.env.observation_space.shape[0])
        mlflow.log_param("action_size", self.env.action_space.n)
        mlflow.log_param("base_model", model_id if model_id else "new")
        
        # Initialize W&B with custom config
        self.wandb_run = init_wandb({
            'config': {
                'state_size': self.env.observation_space.shape[0],
                'action_size': self.env.action_space.n,
                'learning_rate': self.agent.learning_rate,
                'gamma': self.agent.gamma,
                'epsilon_start': self.agent.epsilon,
                'epsilon_min': self.agent.epsilon_min,
                'epsilon_decay': self.agent.epsilon_decay,
                'batch_size': self.agent.batch_size,
                'memory_size': len(self.agent.memory),
                'base_model': model_id if model_id else "new"
            }
        })
        
        # Log model architecture
        if hasattr(self.agent, 'model'):
            log_model_summary(self.agent.model)
        
        return True

    def train(self, episodes=1000, batch_size=32, model_id=None, run_name=None, use_wandb=True, use_mlflow=True):
        """Train the agent"""
        if self.is_training:
            return False, "Training is already in progress"
        
        # Initialize environment and agent if not already done
        if self.env is None or self.agent is None:
            self.initialize_training(model_id)
        
        # Generate a unique model ID for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_id = f"model_{timestamp}"
        
        # Set up MLflow tracking
        if use_mlflow:
            try:
                experiment_name = "flappy-bird-rl"
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run(run_name=run_name or f"training_{timestamp}") as run:
                    # Log parameters
                    mlflow.log_param("episodes", episodes)
                    mlflow.log_param("batch_size", batch_size)
                    mlflow.log_param("base_model", model_id or "new")
                    mlflow.log_param("model_id", new_model_id)
                    
                    # Train the agent and track metrics
                    self._train_loop(episodes, batch_size, new_model_id, run, use_wandb)
                    
                    # Log the final model
                    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{new_model_id}.h5')
                    mlflow.log_artifact(model_path)
            except Exception as e:
                print(f"Error with MLflow tracking: {str(e)}")
                # Continue with training even if MLflow fails
                self._train_loop(episodes, batch_size, new_model_id, None, use_wandb)
        else:
            # Train without MLflow
            self._train_loop(episodes, batch_size, new_model_id, None, use_wandb)
        
        return True, f"Training completed. Model saved as {new_model_id}"

    def _train_loop(self, episodes, batch_size, model_id, mlflow_run=None, use_wandb=True):
        """Internal training loop with visualization"""
        self.is_training = True
        self.episode_rewards = []
        self.episode_scores = []
        
        try:
            for e in range(1, episodes + 1):
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.agent.state_size])
                done = False
                score = 0
                reward_sum = 0
                step = 0
                
                while not done:
                    # Get action
                    action = self.agent.act(state)
                    
                    # Take action
                    next_state, reward, done, _, info = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.agent.state_size])
                    
                    # Render frame and emit to frontend if socketio is available
                    if self.socketio:
                        frame = self.env.render()
                        if frame is not None:
                            # Encode the frame as base64
                            frame_base64 = base64.b64encode(frame).decode('utf-8')
                            self.socketio.emit('game_frame', {
                                'frame': frame_base64,
                                'score': info.get('score', 0),
                                'episode': e,
                                'epsilon': self.agent.epsilon
                            })
                            # Slow down rendering to make it visible (30 FPS)
                            time.sleep(0.033)
                    
                    # Store experience
                    self.agent.remember(state[0], action, reward, next_state[0], done)
                    
                    # Update state and metrics
                    state = next_state
                    reward_sum += reward
                    score = info.get('score', 0)
                    step += 1
                    
                    # Emit step update
                    if self.socketio and step % 10 == 0:
                        self.socketio.emit('training_step', {
                            'episode': e,
                            'step': step,
                            'score': score,
                            'reward': reward_sum
                        })
                
                # Train the agent
                if len(self.agent.memory) > batch_size:
                    self.agent.replay(batch_size)
                
                # Update target network periodically
                if e % self.agent.update_target_frequency == 0:
                    self.agent.update_target_model()
                
                # Track metrics
                self.episode_rewards.append(reward_sum)
                self.episode_scores.append(score)
                
                # Update best score
                if score > self.best_score:
                    self.best_score = score
                
                # Log metrics
                if mlflow_run:
                    mlflow.log_metric("reward", reward_sum, step=e)
                    mlflow.log_metric("score", score, step=e)
                    mlflow.log_metric("epsilon", self.agent.epsilon, step=e)
                
                if use_wandb and self.wandb_run:
                    log_episode_metrics(self.wandb_run, e, score, reward_sum, self.agent.epsilon)
                
                # Emit episode update
                if self.socketio:
                    self.socketio.emit('training_progress', {
                        'episode': e,
                        'total_episodes': episodes,
                        'score': score,
                        'reward': reward_sum,
                        'best_score': self.best_score,
                        'epsilon': self.agent.epsilon
                    })
                
                # Print progress
                print(f"Episode: {e}/{episodes}, Score: {score}, Reward: {reward_sum:.2f}, Epsilon: {self.agent.epsilon:.4f}")
                
                # Save model periodically
                if e % 100 == 0 or e == episodes:
                    self.agent.save(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{model_id}.h5'))
            
            # Save final model
            self.agent.save(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{model_id}.h5'))
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
        finally:
            self.is_training = False
            
        return self.episode_rewards, self.episode_scores

    def analyze_training_data(self, data):
        """Analyze training data using pandas instead of PySpark"""
        try:
            import pandas as pd
            
            # Convert data to pandas DataFrame
            df = pd.DataFrame(data, columns=["episode", "reward", "score"])
            
            # Calculate statistics
            stats = {
                "total_episodes": len(df),
                "avg_reward": df["reward"].mean(),
                "avg_score": df["score"].mean(),
                "max_score": df["score"].max(),
                "reward_trend": df["reward"].rolling(window=10).mean().tolist(),
                "score_trend": df["score"].rolling(window=10).mean().tolist()
            }
            
            # Calculate correlation
            correlation = df[["reward", "score"]].corr().to_dict()
            
            return {
                "stats": stats,
                "correlation": correlation
            }
        except Exception as e:
            print(f"Error analyzing training data: {str(e)}")
            return {"error": str(e)}


if __name__ == "__main__":
    # Example usage
    manager = TrainingManager()
    manager.train(episodes=100, save_interval=25, use_mlflow=True, use_wandb=False) 