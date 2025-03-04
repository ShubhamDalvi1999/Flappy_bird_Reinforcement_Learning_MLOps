import { useState, useEffect } from 'react';
import Head from 'next/head';
import axios from 'axios';
import GameComponent from '../components/GameComponent';
import ModelMetrics from '../components/ModelMetrics';
import MLflowExperiments from '../components/MLflowExperiments';
import WandBDashboard from '../components/WandBDashboard';
import TrainingVisualization from '../components/TrainingVisualization';

// Use a relative URL for API calls
const API_BASE_URL = '/api';

export default function Home() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState('');
  const [activeTab, setActiveTab] = useState('game');
  const [showTrainingViz, setShowTrainingViz] = useState(false);

  useEffect(() => {
    // Fetch models on load
    fetchModels();
    
    // Check if training is in progress
    checkTrainingStatus();
    
    // Poll training status every 5 seconds if training is in progress
    const interval = setInterval(() => {
      if (isTraining) {
        checkTrainingStatus();
      }
    }, 5000);
    
    return () => clearInterval(interval);
  }, [isTraining]);

  const fetchModels = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      if (response.data.status === 'success') {
        setModels(response.data.models);
        
        // Select the first model by default if available
        if (response.data.models.length > 0 && !selectedModel) {
          setSelectedModel(response.data.models[0].id);
        }
      }
    } catch (error) {
      console.error('Error fetching models:', error);
    }
  };

  const checkTrainingStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/training/status`);
      if (response.data.status === 'success') {
        setIsTraining(response.data.is_training);
        setTrainingStatus(response.data.message);
        
        // If training just completed, refresh models list
        if (!response.data.is_training && isTraining) {
          fetchModels();
        }
      }
    } catch (error) {
      console.error('Error checking training status:', error);
    }
  };

  const startTraining = async () => {
    try {
      setIsTraining(true);
      setTrainingStatus('Starting training...');
      setShowTrainingViz(true);
      
      const response = await axios.post(`${API_BASE_URL}/training/start`, {
        episodes: 1000,
        save_interval: 100
      });
      
      if (response.data.status === 'success') {
        setTrainingStatus(response.data.message);
      } else {
        setTrainingStatus('Failed to start training');
        setIsTraining(false);
      }
    } catch (error) {
      console.error('Error starting training:', error);
      setTrainingStatus('Error starting training');
      setIsTraining(false);
    }
  };

  const stopTraining = async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/training/stop`);
      
      if (response.data.status === 'success') {
        setTrainingStatus(response.data.message);
        // Don't immediately set isTraining to false
        // Wait for the status check to confirm training has stopped
      } else {
        setTrainingStatus(`Failed to stop training: ${response.data.message}`);
      }
    } catch (error) {
      console.error('Error stopping training:', error);
      setTrainingStatus('Error stopping training');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>Flappy Bird RL - Production Ready</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-8">
            <h1 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">
              Flappy Bird Reinforcement Learning
            </h1>
            <p className="mt-3 max-w-2xl mx-auto text-xl text-gray-500 sm:mt-4">
              A production-grade AI application using MLflow, Weights & Biases, and PySpark
            </p>
          </div>

          {/* Tab navigation */}
          <div className="border-b border-gray-200 mb-8">
            <nav className="-mb-px flex space-x-8">
              <button
                onClick={() => setActiveTab('game')}
                className={`${
                  activeTab === 'game'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                Game
              </button>
              <button
                onClick={() => setActiveTab('metrics')}
                className={`${
                  activeTab === 'metrics'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                Model Metrics
              </button>
              <button
                onClick={() => setActiveTab('experiments')}
                className={`${
                  activeTab === 'experiments'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                MLflow Experiments
              </button>
              <button
                onClick={() => setActiveTab('wandb')}
                className={`${
                  activeTab === 'wandb'
                    ? 'border-indigo-500 text-indigo-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
              >
                Weights & Biases
              </button>
            </nav>
          </div>

          {/* Tab content */}
          <div>
            {activeTab === 'game' && (
              <div className="grid grid-cols-1 gap-y-8 lg:grid-cols-3 lg:gap-x-8">
                <div className="lg:col-span-2">
                  <div className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="p-6">
                      <h2 className="text-lg font-medium text-gray-900 mb-4">
                        {showTrainingViz ? "Training Visualization" : "Game Simulation"}
                      </h2>
                      
                      {showTrainingViz ? (
                        <TrainingVisualization />
                      ) : (
                        <>
                          <div className="flex items-center mb-4">
                            <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mr-4">
                              Select Model:
                            </label>
                            <select
                              id="model-select"
                              className="block w-full max-w-xs border-gray-300 rounded-md shadow-sm p-2 border"
                              value={selectedModel}
                              onChange={(e) => setSelectedModel(e.target.value)}
                            >
                              {models.map((model) => (
                                <option key={model.id} value={model.id}>
                                  {model.name} ({new Date(model.created_at).toLocaleDateString()})
                                </option>
                              ))}
                            </select>
                          </div>
                          
                          <GameComponent selectedModel={selectedModel} />
                        </>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="lg:col-span-1">
                  <div className="bg-white overflow-hidden shadow rounded-lg">
                    <div className="p-6">
                      <h2 className="text-lg font-medium text-gray-900 mb-4">Training Controls</h2>
                      
                      <div className="mb-4">
                        {isTraining ? (
                          <button
                            onClick={stopTraining}
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                          >
                            Stop Training
                          </button>
                        ) : (
                          <button
                            onClick={startTraining}
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                          >
                            Start Training
                          </button>
                        )}
                        
                        {!isTraining && showTrainingViz && (
                          <button
                            onClick={() => setShowTrainingViz(false)}
                            className="ml-4 inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                          >
                            Show Game
                          </button>
                        )}
                        
                        {!isTraining && !showTrainingViz && (
                          <button
                            onClick={() => setShowTrainingViz(true)}
                            className="ml-4 inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                          >
                            Show Training Viz
                          </button>
                        )}
                      </div>
                      
                      <div className="mt-4">
                        <h3 className="text-sm font-medium text-gray-700">Training Status:</h3>
                        <div className="mt-2 max-w-xl text-sm text-gray-500">
                          {isTraining ? (
                            <div className="flex items-center">
                              <div className="mr-2 h-4 w-4 bg-indigo-600 rounded-full animate-pulse"></div>
                              <p>{trainingStatus}</p>
                            </div>
                          ) : (
                            <p>{trainingStatus || 'Not training'}</p>
                          )}
                        </div>
                      </div>
                      
                      <div className="mt-6">
                        <h3 className="text-sm font-medium text-gray-700">Available Models:</h3>
                        <ul className="mt-2 divide-y divide-gray-200">
                          {models.map((model) => (
                            <li key={model.id} className="py-2">
                              <div className="flex items-center justify-between">
                                <div>
                                  <p className="text-sm font-medium text-gray-900">{model.name}</p>
                                  <p className="text-sm text-gray-500">Created: {new Date(model.created_at).toLocaleDateString()}</p>
                                </div>
                                <span 
                                  className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                    model.id === selectedModel
                                      ? 'bg-green-100 text-green-800'
                                      : 'bg-gray-100 text-gray-800'
                                  }`}
                                >
                                  {model.id === selectedModel ? 'Selected' : 'Select'}
                                </span>
                              </div>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === 'metrics' && <ModelMetrics />}
            
            {activeTab === 'experiments' && <MLflowExperiments />}
            
            {activeTab === 'wandb' && <WandBDashboard />}
          </div>
        </div>
      </main>

      <footer className="bg-white">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            Flappy Bird Reinforcement Learning - Production Grade AI Application with MLflow, W&B and PySpark
          </p>
        </div>
      </footer>
    </div>
  );
} 