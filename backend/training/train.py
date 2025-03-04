import os
import time
import json
import numpy as np
import torch
import mlflow
import wandb
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from game.flappy_bird import FlappyBirdEnv
from agent.dqn_agent import DQNAgent

class TrainingManager:
    def __init__(self):
        self.env = None
        self.agent = None
        self.is_training = False
        self.stop_requested = False
        self.current_episode = 0
        self.total_episodes = 0
        self.latest_score = 0
        self.best_score = 0
        self.training_stats = []
        self.message = "Not training"
        self.spark = None
        
    def train(self, episodes=1000, batch_size=64, save_interval=100, use_mlflow=True, use_wandb=True):
        """
        Train the agent
        
        Args:
            episodes (int): Number of episodes to train
            batch_size (int): Batch size for learning
            save_interval (int): Save model every N episodes
            use_mlflow (bool): Whether to log to MLflow
            use_wandb (bool): Whether to log to Weights & Biases
        """
        if self.is_training:
            return {"status": "error", "message": "Training already in progress"}
        
        self.is_training = True
        self.stop_requested = False
        self.current_episode = 0
        self.total_episodes = episodes
        self.training_stats = []
        self.message = "Training started"
        
        try:
            # Initialize environment and agent
            self.env = FlappyBirdEnv(render=False)
            self.agent = DQNAgent(
                state_size=self.env.state_size,
                action_size=self.env.action_size,
                batch_size=batch_size
            )
            
            # Initialize MLflow
            if use_mlflow:
                mlflow.set_experiment("flappy_bird_rl")
                mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                mlflow.log_param("episodes", episodes)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", self.agent.learning_rate)
                mlflow.log_param("gamma", self.agent.gamma)
                mlflow.log_param("hidden_sizes", str(self.agent.hidden_sizes))
            
            # Initialize W&B
            if use_wandb:
                wandb.init(
                    project="flappy_bird_rl",
                    name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "episodes": episodes,
                        "batch_size": batch_size,
                        "learning_rate": self.agent.learning_rate,
                        "gamma": self.agent.gamma,
                        "hidden_sizes": self.agent.hidden_sizes
                    }
                )
            
            # Training loop
            scores = []
            epsilons = []
            losses = []
            
            for episode in range(1, episodes + 1):
                if self.stop_requested:
                    self.message = "Training stopped by user"
                    break
                
                self.current_episode = episode
                state = self.env.reset()
                score = 0
                done = False
                
                while not done:
                    # Get agent's action
                    action = self.agent.act(state)
                    
                    # Take action in environment
                    next_state, reward, done, _ = self.env.step(action)
                    
                    # Store experience in replay memory
                    self.agent.remember(state, action, reward, next_state, done)
                    
                    # Update state and score
                    state = next_state
                    score += reward
                    
                    # If game over, end episode
                    if done:
                        break
                
                # Train the agent after each episode
                loss = self.agent.replay()
                
                # Update epsilon
                self.agent.update_epsilon()
                
                # Store episode results
                scores.append(score)
                epsilons.append(self.agent.epsilon)
                losses.append(loss)
                
                # Update latest and best scores
                self.latest_score = score
                self.best_score = max(self.best_score, score)
                
                # Log metrics
                if use_mlflow:
                    mlflow.log_metric("score", score, step=episode)
                    mlflow.log_metric("epsilon", self.agent.epsilon, step=episode)
                    if loss is not None:
                        mlflow.log_metric("loss", loss, step=episode)
                    mlflow.log_metric("avg_score_last_100", np.mean(scores[-100:]), step=episode)
                
                if use_wandb:
                    wandb.log({
                        "episode": episode,
                        "score": score,
                        "epsilon": self.agent.epsilon,
                        "loss": loss if loss is not None else 0,
                        "avg_score_last_100": np.mean(scores[-100:])
                    })
                
                # Save progress data
                self.training_stats.append({
                    "episode": episode,
                    "score": score,
                    "epsilon": self.agent.epsilon,
                    "loss": loss if loss is not None else 0,
                    "avg_score_last_100": np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                })
                
                # Print progress
                if episode % 10 == 0:
                    avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
                    self.message = f"Episode: {episode}/{episodes} | Score: {score:.2f} | Avg Score: {avg_score:.2f} | Epsilon: {self.agent.epsilon:.4f}"
                    print(self.message)
                
                # Save model
                if episode % save_interval == 0 or episode == episodes:
                    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = os.path.join(model_dir, f"dqn_model_ep{episode}.pth")
                    self.agent.save(model_path)
                    
                    if use_mlflow:
                        mlflow.log_artifact(model_path)
                        
                    if use_wandb:
                        wandb.save(model_path)
            
            # Save final progress data
            self.save_progress(scores, epsilons, losses)
            
            # Analyze training data with Spark
            if len(scores) > 100:
                self.analyze_with_spark(scores, epsilons, losses)
            
            # End logging sessions
            if use_mlflow:
                mlflow.end_run()
            
            if use_wandb:
                wandb.finish()
            
            self.message = "Training completed"
            
        except Exception as e:
            self.message = f"Training error: {str(e)}"
            print(f"Error during training: {str(e)}")
        
        finally:
            self.is_training = False
            if self.env:
                self.env.close()
    
    def stop_training(self):
        """Request to stop the training process"""
        self.stop_requested = True
        return {"status": "success", "message": "Stop requested"}
    
    def get_status(self):
        """Get current training status"""
        progress = {}
        if len(self.training_stats) > 0:
            recent_stats = self.training_stats[-min(100, len(self.training_stats)):]
            progress = {
                "episodes": [s["episode"] for s in recent_stats],
                "scores": [s["score"] for s in recent_stats],
                "avg_scores": [s["avg_score_last_100"] for s in recent_stats],
                "epsilons": [s["epsilon"] for s in recent_stats],
                "losses": [s["loss"] for s in recent_stats]
            }
        
        return {
            "is_training": self.is_training,
            "message": self.message,
            "current_episode": self.current_episode,
            "total_episodes": self.total_episodes,
            "latest_score": self.latest_score,
            "best_score": self.best_score,
            "progress": progress
        }
    
    def save_progress(self, scores, epsilons, losses):
        """Save training progress to file"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        progress_path = os.path.join(data_dir, f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        progress_data = {
            "episodes": list(range(1, len(scores) + 1)),
            "scores": scores,
            "epsilons": epsilons,
            "losses": [l if l is not None else 0 for l in losses],
            "timestamp": time.time()
        }
        
        with open(progress_path, "w") as f:
            json.dump(progress_data, f)
    
    def init_spark(self):
        """Initialize Spark session"""
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName("FlappyBirdAnalytics") \
                .config("spark.executor.memory", "2g") \
                .config("spark.driver.memory", "2g") \
                .getOrCreate()
        return self.spark
    
    def analyze_with_spark(self, scores, epsilons, losses):
        """Analyze training data with Spark"""
        try:
            # Initialize Spark
            spark = self.init_spark()
            
            # Create dataframe
            data = [{
                "episode": i + 1,
                "score": scores[i],
                "epsilon": epsilons[i],
                "loss": losses[i] if losses[i] is not None else 0
            } for i in range(len(scores))]
            
            df = spark.createDataFrame(data)
            
            # Calculate rolling average
            window_size = 100
            window_spec = Window.orderBy("episode").rowsBetween(-(window_size-1), 0)
            df = df.withColumn("avg_score_100", F.avg("score").over(window_spec))
            
            # Calculate correlation
            assembler = VectorAssembler(
                inputCols=["score", "epsilon", "loss"],
                outputCol="features"
            )
            df_assembled = assembler.transform(df)
            
            # Compute correlation matrix
            correlation_matrix = Correlation.corr(df_assembled, "features").collect()[0][0]
            print("Correlation Matrix:")
            print(correlation_matrix)
            
            # Log statistics with MLflow
            mlflow.log_metric("correlation_score_epsilon", float(correlation_matrix[0, 1]))
            mlflow.log_metric("correlation_score_loss", float(correlation_matrix[0, 2]))
            
            # More advanced analytics could be added here
            
        except Exception as e:
            print(f"Error during Spark analysis: {str(e)}")


if __name__ == "__main__":
    # Example usage
    manager = TrainingManager()
    manager.train(episodes=100, save_interval=25, use_mlflow=True, use_wandb=False) 