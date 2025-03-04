import numpy as np
import time
import json
import os
import psutil
import pandas as pd
from datetime import datetime
from environment import env, state_size, action_size
from agent import DQNAgent
import mlflow
import mlflow.tensorflow
import wandb

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

def train(episodes=1000, batch_size=32, render=False, run_name=None, use_wandb=True, use_mlflow=True):
    """Main training loop for the DQN agent with MLflow and Weights & Biases tracking"""
    # Initialize experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"flappy_bird_run_{timestamp}"
    
    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            project="flappy-bird-rl",
            name=run_name,
            config={
                "episodes": episodes,
                "batch_size": batch_size,
                "render": render,
                "architecture": "DQN",
                "environment": "FlappyBird"
            }
        )
    
    # Initialize MLflow
    if use_mlflow:
        mlflow.set_tracking_uri("http://mlflow:5000")  # Use the MLflow service hostname in Docker
        mlflow.set_experiment("flappy-bird-rl")
        mlflow.start_run(run_name=run_name)
        
        # Log hyperparameters
        mlflow.log_params({
            "episodes": episodes,
            "batch_size": batch_size,
            "render": render,
            "architecture": "DQN"
        })
    
    agent = DQNAgent(state_size, action_size)
    
    # Log agent hyperparameters
    if use_mlflow:
        mlflow.log_params({
            "learning_rate": agent.learning_rate,
            "gamma": agent.gamma,
            "epsilon_initial": agent.epsilon,
            "epsilon_min": agent.epsilon_min,
            "epsilon_decay": agent.epsilon_decay
        })
    
    # Training metrics
    scores = []
    avg_scores = []
    epsilons = []
    losses = []
    
    # Timestamp for unique log filenames
    progress_file = f"logs/progress_{timestamp}.json"
    model_file = f"models/dqn_model_{timestamp}.h5"
    csv_file = f"logs/training_metrics_{timestamp}.csv"
    
    best_score = -np.inf
    metrics_df = pd.DataFrame(columns=['episode', 'score', 'avg_score', 'epsilon', 'loss', 'memory_usage', 'cpu_usage'])
    
    # Training loop
    for e in range(1, episodes + 1):
        # Reset environment
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])[0]
        
        # Set render mode if requested
        if render:
            env.render_mode = "human"
        
        # Episode variables
        score = 0
        done = False
        episode_loss = 0
        step_count = 0
        
        # Track time for performance metrics
        start_time = time.time()
        
        # Episode loop
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action and observe next state, reward, etc.
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Reshape next_state for memory storage
            next_state = np.reshape(next_state, [1, state_size])[0]
            
            # Store experience in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            
            # Update score
            score += reward
            step_count += 1
        
        # Duration of episode
        duration = time.time() - start_time
        
        # Train the agent using experience replay
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
            if agent.loss_history:
                episode_loss = agent.loss_history[-1]
        
        # Update target model periodically
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()
        
        # Collect metrics
        scores.append(score)
        current_avg_score = np.mean(scores[-100:])  # Average of last 100 episodes
        avg_scores.append(current_avg_score)
        epsilons.append(agent.get_epsilon())
        
        # Collect loss if available
        if agent.loss_history:
            losses.append(agent.loss_history[-1])
        else:
            losses.append(0)
        
        # Collect system metrics
        memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # in MB
        cpu_usage = psutil.cpu_percent()
        
        # Save metrics for Spark analysis
        metrics_df.loc[len(metrics_df)] = {
            'episode': e,
            'score': score,
            'avg_score': current_avg_score,
            'epsilon': agent.get_epsilon(),
            'loss': episode_loss,
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage
        }
        
        # Log metrics to tracking systems
        if use_wandb:
            wandb.log({
                "episode": e,
                "score": score,
                "avg_score": current_avg_score,
                "epsilon": agent.get_epsilon(),
                "loss": episode_loss,
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "steps_per_episode": step_count,
                "episode_duration": duration
            })
            
        if use_mlflow:
            mlflow.log_metrics({
                "score": score,
                "avg_score": current_avg_score,
                "epsilon": agent.get_epsilon(),
                "loss": episode_loss,
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "steps_per_episode": step_count,
                "episode_duration": duration
            }, step=e)
        
        # Print progress
        print(f"Episode: {e}/{episodes} | Score: {score:.2f} | Avg Score: {current_avg_score:.2f} | Epsilon: {agent.epsilon:.4f} | Duration: {duration:.2f}s")
        
        # Save best model
        if current_avg_score > best_score and e > 100:
            best_score = current_avg_score
            agent.save(model_file)
            
            # Log the model in MLflow
            if use_mlflow:
                mlflow.tensorflow.log_model(agent.model, "model")
                mlflow.log_artifact(model_file)
                
            print(f"New best model saved with avg score: {best_score:.2f}")
        
        # Save progress after each episode
        progress = {
            'episodes': list(range(1, e + 1)),
            'scores': scores,
            'avg_scores': avg_scores,
            'epsilons': epsilons,
            'losses': losses,
            'best_score': best_score
        }
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
    
    # Save metrics to CSV for Spark analysis
    metrics_df.to_csv(csv_file, index=False)
    
    # Close tracking systems
    if use_wandb:
        wandb.finish()
        
    if use_mlflow:
        # Log the final metrics CSV as an artifact
        mlflow.log_artifact(csv_file)
        mlflow.end_run()
    
    # Close environment
    env.close()
    
    return progress

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a DQN agent to play Flappy Bird')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for experience replay')
    parser.add_argument('--render', action='store_true', help='Render the game during training')
    parser.add_argument('--run_name', type=str, default=None, help='Name for this training run')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable Weights & Biases tracking')
    parser.add_argument('--disable_mlflow', action='store_true', help='Disable MLflow tracking')
    
    args = parser.parse_args()
    
    # Train the agent
    progress = train(
        episodes=args.episodes, 
        batch_size=args.batch_size, 
        render=args.render,
        run_name=args.run_name,
        use_wandb=not args.disable_wandb,
        use_mlflow=not args.disable_mlflow
    )
    
    print("Training complete!") 