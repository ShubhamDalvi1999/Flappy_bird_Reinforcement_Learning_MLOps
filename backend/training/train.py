import os
import time
import json
import numpy as np
from datetime import datetime
from flask_socketio import SocketIO
import base64

from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent

class TrainingManager:
    def __init__(self, socketio=None):
        self.env = None
        self.agent = None
        self.is_training = False
        self.socketio = socketio
        self.episode_rewards = []
        self.episode_scores = []
        self.best_score = 0
        self.current_model_id = None
        self.current_episode = 0

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
        return True

    def train(self, episodes=1000, batch_size=32, model_id=None, run_name=None, use_wandb=False, use_mlflow=False):
        """Train the agent"""
        if self.is_training:
            return False, "Training is already in progress"
        
        # Initialize environment and agent if not already done
        if self.env is None or self.agent is None:
            self.initialize_training(model_id)
        
        # Generate a unique model ID for this training run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_id = f"model_{timestamp}"
        
        # Reset training metrics
        self.episode_rewards = []
        self.episode_scores = []
        self.best_score = 0
        
        # Start training
        self.is_training = True
        print(f"Starting training for {episodes} episodes")
        
        try:
            for episode in range(episodes):
                if not self.is_training:
                    print("Training stopped by user")
                    break
                
                self.current_episode = episode + 1
                state, _ = self.env.reset()
                state = np.reshape(state, [1, self.env.observation_space.shape[0]])
                done = False
                score = 0
                total_reward = 0
                
                # For visualization
                frame_count = 0
                
                while not done:
                    # Get action from agent
                    action = self.agent.act(state[0])
                    
                    # Take action
                    next_state, reward, done, _, info = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                    
                    # Store experience in memory
                    self.agent.remember(state[0], action, reward, next_state[0], done)
                    
                    # Update state
                    state = next_state
                    
                    # Update metrics
                    total_reward += reward
                    score = info.get('score', 0)
                    
                    # Train agent
                    loss = self.agent.replay(batch_size)
                    
                    # Update target network periodically
                    if frame_count % self.agent.update_target_frequency == 0:
                        self.agent.update_target_model()
                    
                    frame_count += 1
                    
                    # Send frame to client every 5 frames for visualization
                    if self.socketio and frame_count % 5 == 0:
                        frame = self.env.render()
                        self.socketio.emit('game_frame', {
                            'frame': base64.b64encode(frame).decode('utf-8'),
                            'score': score,
                            'episode': episode + 1,
                            'epsilon': self.agent.epsilon
                        })
                
                # Update best score
                if score > self.best_score:
                    self.best_score = score
                
                # Store episode metrics
                self.episode_rewards.append(total_reward)
                self.episode_scores.append(score)
                
                # Log episode metrics
                print(f"Episode: {episode+1}/{episodes}, Score: {score}, Reward: {total_reward:.2f}, Epsilon: {self.agent.epsilon:.4f}")
                
                # Send training progress to client
                if self.socketio:
                    self.socketio.emit('training_progress', {
                        'episode': episode + 1,
                        'total_episodes': episodes,
                        'score': score,
                        'reward': total_reward,
                        'best_score': self.best_score,
                        'epsilon': self.agent.epsilon
                    })
                
                # Save model periodically
                if (episode + 1) % 100 == 0:
                    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{new_model_id}.h5')
                    self.agent.save(model_path)
                    print(f"Model saved to {model_path}")
            
            # Save final model
            final_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', f'{new_model_id}_final.h5')
            self.agent.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
            
            self.is_training = False
            return True, f"Training completed for {episodes} episodes"
        
        except Exception as e:
            self.is_training = False
            print(f"Error during training: {e}")
            return False, f"Error during training: {str(e)}"

    def stop_training(self):
        """Stop the training process"""
        self.is_training = False
        return True

    def get_training_stats(self):
        """Get training statistics"""
        if not self.episode_rewards:
            return None
        
        return {
            'episode_count': len(self.episode_rewards),
            'best_score': self.best_score,
            'average_score': sum(self.episode_scores) / len(self.episode_scores),
            'average_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'latest_score': self.episode_scores[-1],
            'latest_reward': self.episode_rewards[-1],
            'scores': self.episode_scores,
            'rewards': self.episode_rewards
        }


if __name__ == "__main__":
    # Example usage
    manager = TrainingManager()
    manager.train(episodes=100, save_interval=25, use_mlflow=True, use_wandb=False) 