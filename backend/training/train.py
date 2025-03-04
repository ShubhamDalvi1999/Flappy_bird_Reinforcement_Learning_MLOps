import os
import time
import json
import numpy as np
import torch
import mlflow
import wandb
from datetime import datetime
from flask_socketio import SocketIO
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
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

    def train(self, num_episodes=1000, model_id=None, config=None):
        """Train the agent with enhanced configuration and monitoring"""
        if not self.env or not self.agent or (model_id and model_id != self.current_model_id):
            if not self.initialize_training(model_id):
                raise Exception("Failed to initialize training")
        
        self.is_training = True
        start_time = time.time()
        total_steps = 0
        best_score = float('-inf')
        episodes_without_improvement = 0
        
        # Apply configuration
        if config:
            if config.get('learning_rate'):
                self.agent.learning_rate = config['learning_rate']
            if config.get('batch_size'):
                self.agent.batch_size = config['batch_size']
            if config.get('epsilon_decay'):
                self.agent.epsilon_decay = config['epsilon_decay']
        
        try:
            # Generate a unique model ID for this training run
            new_model_id = f"model_{int(time.time())}"
            
            # Start MLflow run with proper tags
            with mlflow.start_run() as run:
                # Log model parameters
                mlflow.log_params({
                    'learning_rate': self.agent.learning_rate,
                    'gamma': self.agent.gamma,
                    'epsilon_start': self.agent.epsilon,
                    'epsilon_min': self.agent.epsilon_min,
                    'epsilon_decay': self.agent.epsilon_decay,
                    'batch_size': self.agent.batch_size,
                    'memory_size': len(self.agent.memory),
                    'architecture': 'DQN',
                    'base_model': model_id if model_id else "new",
                    **{k: v for k, v in (config or {}).items() if v is not None}
                })
                
                # Tag the run with model ID
                mlflow.set_tag('model_id', new_model_id)
                
                checkpoint_interval = config.get('checkpoint_interval', 100) if config else 100
                target_score = config.get('target_score') if config else None
                early_stopping = config.get('early_stopping', True) if config else True
                
                for episode in range(num_episodes):
                    if not self.is_training:
                        break
                        
                    state, _ = self.env.reset()
                    total_reward = 0
                    done = False
                    episode_loss = []
                    episode_steps = 0
                    
                    while not done and self.is_training:
                        # Get action from agent
                        action = self.agent.get_action(state)
                        
                        # Take action in environment
                        next_state, reward, done, _, _ = self.env.step(action)
                        
                        # Store experience in replay memory
                        self.agent.remember(state, action, reward, next_state, done)
                        
                        # Update state and accumulate reward
                        state = next_state
                        total_reward += reward
                        episode_steps += 1
                        total_steps += 1
                        
                        # Render the game and emit frame
                        if self.socketio:
                            frame = self.env.render()
                            frame_base64 = base64.b64encode(frame).decode('utf-8')
                            self.socketio.emit('game_frame', {
                                'frame': frame_base64,
                                'score': self.env.score,
                                'episode': episode + 1,
                                'epsilon': self.agent.epsilon,
                                'total_episodes': num_episodes,
                                'estimated_time_remaining': (num_episodes - episode) * (time.time() - start_time) / (episode + 1) if episode > 0 else None
                            })
                            time.sleep(0.033)  # ~30 FPS
                        
                        # Train the agent
                        if len(self.agent.memory) > self.agent.batch_size:
                            loss = self.agent.replay(self.agent.batch_size)
                            if loss is not None:
                                episode_loss.append(loss)
                    
                    # Calculate average loss for the episode
                    avg_loss = np.mean(episode_loss) if episode_loss else None
                    
                    # Update best score and check for early stopping
                    if self.env.score > best_score:
                        best_score = self.env.score
                        episodes_without_improvement = 0
                        
                        # Save best model
                        model_path = f"models/best_model_{best_score}.h5"
                        self.agent.save_model(model_path)
                        mlflow.log_artifact(model_path)
                        if self.wandb_run:
                            wandb.save(model_path)
                    else:
                        episodes_without_improvement += 1
                    
                    # Early stopping check
                    if early_stopping and episodes_without_improvement >= 50:  # No improvement in 50 episodes
                        print("Early stopping triggered - No improvement in 50 episodes")
                        if self.socketio:
                            self.socketio.emit('training_info', {
                                'message': 'Early stopping triggered - No improvement in 50 episodes'
                            })
                        break
                    
                    # Target score check
                    if target_score and self.env.score >= target_score:
                        print(f"Target score {target_score} achieved!")
                        if self.socketio:
                            self.socketio.emit('training_info', {
                                'message': f'Target score {target_score} achieved!'
                            })
                        break
                    
                    # Checkpoint saving
                    if (episode + 1) % checkpoint_interval == 0:
                        checkpoint_path = f"models/checkpoint_{new_model_id}_ep{episode + 1}.h5"
                        self.agent.save_model(checkpoint_path)
                        mlflow.log_artifact(checkpoint_path)
                        if self.wandb_run:
                            wandb.save(checkpoint_path)
                    
                    # Log metrics and emit progress
                    metrics = {
                        'episode_reward': total_reward,
                        'score': self.env.score,
                        'epsilon': self.agent.epsilon,
                        'best_score': best_score,
                        'episode_steps': episode_steps,
                        'total_steps': total_steps,
                        'average_score': np.mean(self.episode_scores[-100:] if self.episode_scores else [0]),
                        'average_reward': np.mean(self.episode_rewards[-100:] if self.episode_rewards else [0])
                    }
                    if avg_loss is not None:
                        metrics['loss'] = avg_loss
                    
                    mlflow.log_metrics(metrics, step=episode)
                    
                    if self.wandb_run:
                        log_episode_metrics(
                            episode=episode + 1,
                            score=self.env.score,
                            reward=total_reward,
                            epsilon=self.agent.epsilon,
                            loss=avg_loss
                        )
                    
                    # Store episode data
                    self.episode_rewards.append(total_reward)
                    self.episode_scores.append(self.env.score)
                    
                    # Emit detailed training stats
                    if self.socketio:
                        self.socketio.emit('training_stats', {
                            'episode': episode + 1,
                            'total_episodes': num_episodes,
                            'score': self.env.score,
                            'reward': total_reward,
                            'epsilon': self.agent.epsilon,
                            'best_score': best_score,
                            'loss': avg_loss,
                            'estimated_time_remaining': (num_episodes - episode) * (time.time() - start_time) / (episode + 1) if episode > 0 else None,
                            'episodes_without_improvement': episodes_without_improvement
                        })
                    
                    print(f"Episode: {episode + 1}/{num_episodes}, Score: {self.env.score}, "
                          f"Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.2f}"
                          + (f", Loss: {avg_loss:.4f}" if avg_loss is not None else ""))
                
                # Log final metrics
                training_time = (time.time() - start_time) / 60  # Convert to minutes
                final_metrics = {
                    'final_epsilon': self.agent.epsilon,
                    'max_score': best_score,
                    'training_time_minutes': training_time,
                    'total_episodes': episode + 1,
                    'total_steps': total_steps,
                    'final_average_score': np.mean(self.episode_scores[-100:]),
                    'final_average_reward': np.mean(self.episode_rewards[-100:]),
                    'early_stopped': episodes_without_improvement >= 50 if early_stopping else False
                }
                mlflow.log_metrics(final_metrics)
            
            # Save the final model
            final_model_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'models',
                f'{new_model_id}.h5'
            )
            self.agent.save(final_model_path)
            mlflow.log_artifact(final_model_path)
            
            if self.wandb_run:
                wandb.save(final_model_path)
            
            return new_model_id
            
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            if self.socketio:
                self.socketio.emit('training_error', {'error': str(e)})
            raise e
        
        finally:
            self.stop_training()
            if self.wandb_run:
                # Log final summary data
                wandb.run.summary.update({
                    'final_epsilon': self.agent.epsilon,
                    'best_score': best_score,
                    'final_memory_size': len(self.agent.memory),
                    'total_episodes': episode + 1 if 'episode' in locals() else 0,
                    'average_score': np.mean(self.episode_scores),
                    'average_reward': np.mean(self.episode_rewards),
                    'training_time_minutes': (time.time() - start_time) / 60
                })
                wandb.finish()

    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        if self.env:
            self.env.close()

    def get_training_stats(self):
        """Get current training statistics"""
        if not self.episode_rewards:
            return None
            
        return {
            'total_episodes': len(self.episode_rewards),
            'latest_reward': self.episode_rewards[-1],
            'latest_score': self.episode_scores[-1],
            'best_score': self.best_score,
            'average_reward': np.mean(self.episode_rewards[-100:]),
            'average_score': np.mean(self.episode_scores[-100:])
        }

    def analyze_training_data(self):
        """Analyze training data using PySpark"""
        if not self.episode_rewards:
            return None
            
        # Create SparkSession
        spark = SparkSession.builder.appName("FlappyBirdAnalysis").getOrCreate()
        
        # Create DataFrame with training data
        data = [(i, r, s) for i, (r, s) in enumerate(zip(self.episode_rewards, self.episode_scores))]
        df = spark.createDataFrame(data, ["episode", "reward", "score"])
        
        # Calculate correlations
        assembler = VectorAssembler(inputCols=["reward", "score"], outputCol="features")
        df_vector = assembler.transform(df)
        correlation = Correlation.corr(df_vector, "features").head()[0].toArray()
        
        # Calculate statistics
        stats = df.select(
            [
                "reward",
                "score"
            ]
        ).summary("count", "mean", "stddev", "min", "max").toPandas()
        
        spark.stop()
        
        return {
            'correlation': correlation.tolist(),
            'stats': json.loads(stats.to_json())
        }


if __name__ == "__main__":
    # Example usage
    manager = TrainingManager()
    manager.train(episodes=100, save_interval=25, use_mlflow=True, use_wandb=False) 