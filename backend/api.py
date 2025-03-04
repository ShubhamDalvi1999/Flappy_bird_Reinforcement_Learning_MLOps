from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import subprocess
import json
import os
import threading
import time
import glob
import pandas as pd
import requests
from pyspark.sql import SparkSession
import mlflow
from mlflow.tracking import MlflowClient

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable to track if training is running
training_process = None
training_thread = None
training_status = {
    "running": False,
    "episodes": 0,
    "current_episode": 0,
    "latest_score": 0,
    "best_score": 0
}

# Initialize Spark session - will be lazy-loaded when needed
spark = None

def get_spark():
    """Get or create Spark session"""
    global spark
    if spark is None:
        spark = SparkSession.builder \
            .appName("FlappyBirdAnalysis") \
            .config("spark.driver.memory", "1g") \
            .config("spark.executor.memory", "1g") \
            .getOrCreate()
    return spark

def get_latest_progress_file():
    """Get the most recent progress file from logs directory"""
    try:
        progress_files = glob.glob('logs/progress_*.json')
        if not progress_files:
            return None
        
        # Get the most recent file based on modification time
        latest_file = max(progress_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error getting latest progress file: {e}")
        return None

def get_latest_metrics_file():
    """Get the most recent metrics CSV file from logs directory"""
    try:
        metrics_files = glob.glob('logs/training_metrics_*.csv')
        if not metrics_files:
            return None
        
        # Get the most recent file based on modification time
        latest_file = max(metrics_files, key=os.path.getmtime)
        return latest_file
    except Exception as e:
        print(f"Error getting latest metrics file: {e}")
        return None

def run_training(episodes=1000, batch_size=32, render=False, run_name=None):
    """Run training in a separate thread"""
    global training_status
    
    try:
        # Import here to avoid circular imports
        from train import train
        
        # Update status
        training_status["running"] = True
        training_status["episodes"] = episodes
        training_status["current_episode"] = 0
        
        # Start training
        progress = train(
            episodes=episodes, 
            batch_size=batch_size, 
            render=render,
            run_name=run_name
        )
        
        # Update status when complete
        training_status["running"] = False
        training_status["current_episode"] = episodes
        
        return progress
    except Exception as e:
        print(f"Error in training thread: {e}")
        training_status["running"] = False
        return None

@app.route('/api/start_training', methods=['POST'])
def start_training():
    """Start training the agent"""
    global training_thread, training_status
    
    # Check if training is already running
    if training_status["running"]:
        return jsonify({
            "status": "error", 
            "message": "Training is already running"
        }), 400
    
    # Get parameters from request
    data = request.json or {}
    episodes = data.get('episodes', 1000)
    batch_size = data.get('batch_size', 32)
    render = data.get('render', False)
    run_name = data.get('run_name', None)
    
    # Start training in a separate thread
    training_thread = threading.Thread(
        target=run_training,
        args=(episodes, batch_size, render, run_name)
    )
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({
        "status": "success",
        "message": "Training started"
    })

@app.route('/api/get_progress', methods=['GET'])
def get_progress():
    """Get the latest training progress"""
    latest_file = get_latest_progress_file()
    
    if not latest_file:
        return jsonify({
            "status": "no_data",
            "progress": None,
            "training_status": training_status
        })
    
    try:
        with open(latest_file, 'r') as f:
            progress = json.load(f)
            
            # Update training status with latest information from progress file
            if progress and 'scores' in progress:
                training_status["current_episode"] = len(progress['scores'])
                training_status["latest_score"] = progress['scores'][-1]
                training_status["best_score"] = progress.get('best_score', 0)
            
            return jsonify({
                "status": "success",
                "progress": progress,
                "training_status": training_status
            })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e),
            "training_status": training_status
        }), 500

@app.route('/api/stop_training', methods=['POST'])
def stop_training():
    """Stop the training process"""
    global training_status
    
    if not training_status["running"]:
        return jsonify({
            "status": "error",
            "message": "No training is running"
        }), 400
    
    # Set status to not running - the training loop should check this and stop
    training_status["running"] = False
    
    return jsonify({
        "status": "success",
        "message": "Training stop requested"
    })

@app.route('/api/models', methods=['GET'])
def list_models():
    """List all trained models"""
    try:
        models = glob.glob('models/*.h5')
        model_list = []
        
        for model_path in models:
            filename = os.path.basename(model_path)
            created = os.path.getmtime(model_path)
            size = os.path.getsize(model_path)
            
            model_list.append({
                "filename": filename,
                "path": model_path,
                "created": created,
                "size": size
            })
        
        # Sort by creation time (newest first)
        model_list.sort(key=lambda x: x["created"], reverse=True)
        
        return jsonify({
            "status": "success",
            "models": model_list
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/mlflow/experiments', methods=['GET'])
def list_mlflow_experiments():
    """List all MLflow experiments"""
    try:
        # Initialize MLflow client
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = MlflowClient()
        
        # Get all experiments
        experiments = client.search_experiments()
        experiment_list = []
        
        for exp in experiments:
            experiment_list.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "creation_time": exp.creation_time
            })
        
        return jsonify({
            "status": "success",
            "experiments": experiment_list
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/mlflow/runs', methods=['GET'])
def list_mlflow_runs():
    """List all runs for a specific experiment"""
    try:
        # Get experiment ID from query parameter
        experiment_id = request.args.get('experiment_id', '0')
        
        # Initialize MLflow client
        mlflow.set_tracking_uri("http://mlflow:5000")
        client = MlflowClient()
        
        # Get all runs for the experiment
        runs = client.search_runs(experiment_id)
        run_list = []
        
        for run in runs:
            run_list.append({
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params
            })
        
        return jsonify({
            "status": "success",
            "runs": run_list
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/spark/analyze', methods=['GET'])
def analyze_with_spark():
    """Analyze training metrics with Spark"""
    try:
        # Get the latest metrics file
        metrics_file = get_latest_metrics_file()
        if not metrics_file:
            return jsonify({
                "status": "error",
                "message": "No metrics file found"
            }), 400
        
        # Get Spark session
        spark = get_spark()
        
        # Read the metrics file
        df = spark.read.csv(metrics_file, header=True, inferSchema=True)
        
        # Register as temp view
        df.createOrReplaceTempView("training_metrics")
        
        # Perform analysis
        analysis_results = {
            "avg_score_by_episode_range": spark.sql("""
                SELECT 
                    CASE 
                        WHEN episode <= 100 THEN '1-100'
                        WHEN episode <= 200 THEN '101-200'
                        WHEN episode <= 300 THEN '201-300'
                        WHEN episode <= 400 THEN '301-400'
                        ELSE '400+'
                    END as episode_range,
                    AVG(score) as avg_score,
                    AVG(avg_score) as avg_running_score,
                    MIN(epsilon) as min_epsilon,
                    MAX(epsilon) as max_epsilon
                FROM training_metrics
                GROUP BY 
                    CASE 
                        WHEN episode <= 100 THEN '1-100'
                        WHEN episode <= 200 THEN '101-200'
                        WHEN episode <= 300 THEN '201-300'
                        WHEN episode <= 400 THEN '301-400'
                        ELSE '400+'
                    END
                ORDER BY episode_range
            """).toPandas().to_dict(orient='records'),
            
            "performance_metrics": spark.sql("""
                SELECT 
                    AVG(memory_usage) as avg_memory_usage_mb,
                    MAX(memory_usage) as max_memory_usage_mb,
                    AVG(cpu_usage) as avg_cpu_usage_percent,
                    MAX(cpu_usage) as max_cpu_usage_percent
                FROM training_metrics
            """).toPandas().to_dict(orient='records')[0],
            
            "training_progression": spark.sql("""
                SELECT 
                    MIN(score) as min_score,
                    MAX(score) as max_score,
                    AVG(score) as avg_score,
                    STDDEV(score) as stddev_score,
                    MIN(loss) as min_loss,
                    MAX(loss) as max_loss,
                    AVG(loss) as avg_loss
                FROM training_metrics
            """).toPandas().to_dict(orient='records')[0]
        }
        
        return jsonify({
            "status": "success",
            "analysis_results": analysis_results,
            "metrics_file": metrics_file
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/spark/save_parquet', methods=['POST'])
def save_as_parquet():
    """Save the metrics data as Parquet for more efficient analysis"""
    try:
        # Get the latest metrics file
        metrics_file = get_latest_metrics_file()
        if not metrics_file:
            return jsonify({
                "status": "error",
                "message": "No metrics file found"
            }), 400
        
        # Get output path from request
        data = request.json or {}
        output_dir = data.get('output_dir', 'analysis_results')
        
        # Get Spark session
        spark = get_spark()
        
        # Read the metrics file
        df = spark.read.csv(metrics_file, header=True, inferSchema=True)
        
        # Save as Parquet
        parquet_path = f"{output_dir}/metrics.parquet"
        df.write.mode("overwrite").parquet(parquet_path)
        
        return jsonify({
            "status": "success",
            "message": f"Metrics saved as Parquet to {parquet_path}"
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/commentary', methods=['GET'])
def get_commentary():
    """Get commentary from Ollama using Llama 3.2 model (if available)"""
    # Only import subprocess here as it's only needed for this route
    import subprocess
    
    # Get the latest score
    latest_file = get_latest_progress_file()
    if not latest_file:
        return jsonify({
            "status": "error",
            "message": "No training data available"
        }), 400
    
    try:
        with open(latest_file, 'r') as f:
            progress = json.load(f)
            latest_score = progress.get('scores', [0])[-1]
            avg_score = progress.get('avg_scores', [0])[-1]
            epsilon = progress.get('epsilons', [1.0])[-1]
        
        # Try to get commentary from Ollama
        try:
            prompt = f"Comment on this Flappy Bird AI training: The agent scored {latest_score:.2f} in its latest episode. Its average score over the last 100 episodes is {avg_score:.2f}. The current exploration rate (epsilon) is {epsilon:.4f}. What does this tell us about the learning process? Provide a brief, insightful comment about how the AI is doing and what this means."
            
            result = subprocess.run(
                ["ollama", "run", "llama3.2:2b", prompt],
                capture_output=True,
                text=True,
                timeout=10  # Timeout after 10 seconds
            )
            
            commentary = result.stdout.strip()
            
            return jsonify({
                "status": "success",
                "commentary": commentary,
                "source": "ollama"
            })
            
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            # Fallback commentary if Ollama is not available
            if latest_score > avg_score:
                commentary = f"The agent scored {latest_score:.2f}, which is above its recent average of {avg_score:.2f}. This suggests it's making progress."
            else:
                commentary = f"The agent scored {latest_score:.2f}, which is below its recent average of {avg_score:.2f}. It's still exploring with epsilon at {epsilon:.4f}."
                
            return jsonify({
                "status": "success", 
                "commentary": commentary,
                "source": "fallback"
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/logs/<path:filename>')
def serve_log(filename):
    """Serve a specific log file"""
    return send_from_directory('logs', filename)

@app.route('/api/models/<path:filename>')
def serve_model(filename):
    """Serve a specific model file"""
    return send_from_directory('models', filename)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 