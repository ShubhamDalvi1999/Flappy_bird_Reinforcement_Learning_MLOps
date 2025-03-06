import { useState, useEffect } from 'react';
import Head from 'next/head';
import axios from 'axios';
import GameComponent from '../components/GameComponent';
import ModelMetrics from '../components/ModelMetrics';
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
      } else {
        setTrainingStatus('Failed to stop training');
      }
    } catch (error) {
      console.error('Error stopping training:', error);
      setTrainingStatus('Error stopping training');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Head>
        <title>Flappy Bird RL</title>
        <meta name="description" content="Flappy Bird Reinforcement Learning" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center mb-8">Flappy Bird Reinforcement Learning</h1>
        
        {/* Navigation Tabs */}
        <div className="flex justify-center mb-6">
          <div className="flex space-x-2 bg-white rounded-lg shadow-md p-1">
            <button
              className={`px-4 py-2 rounded-md ${activeTab === 'game' ? 'bg-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
              onClick={() => setActiveTab('game')}
            >
              Game
            </button>
            <button
              className={`px-4 py-2 rounded-md ${activeTab === 'training' ? 'bg-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
              onClick={() => setActiveTab('training')}
            >
              Training
            </button>
            <button
              className={`px-4 py-2 rounded-md ${activeTab === 'metrics' ? 'bg-blue-500 text-white' : 'text-gray-700 hover:bg-gray-100'}`}
              onClick={() => setActiveTab('metrics')}
            >
              Metrics
            </button>
          </div>
        </div>
        
        {/* Game Tab */}
        {activeTab === 'game' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Play Game with Trained Agent</h2>
            
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Select Model:</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="">Select a model</option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
            
            <GameComponent selectedModel={selectedModel} />
          </div>
        )}
        
        {/* Training Tab */}
        {activeTab === 'training' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Train Agent</h2>
            
            <div className="mb-4">
              <div className="flex space-x-4 mb-4">
                <button
                  className={`px-4 py-2 rounded-md ${isTraining ? 'bg-gray-400 cursor-not-allowed' : 'bg-blue-500 text-white hover:bg-blue-600'}`}
                  onClick={startTraining}
                  disabled={isTraining}
                >
                  Start Training
                </button>
                <button
                  className={`px-4 py-2 rounded-md ${!isTraining ? 'bg-gray-400 cursor-not-allowed' : 'bg-red-500 text-white hover:bg-red-600'}`}
                  onClick={stopTraining}
                  disabled={!isTraining}
                >
                  Stop Training
                </button>
              </div>
              
              <div className="p-3 bg-gray-100 rounded-md">
                <p className="text-gray-700">Status: {trainingStatus || 'Not training'}</p>
              </div>
            </div>
            
            {showTrainingViz && (
              <div className="mt-6">
                <h3 className="text-xl font-semibold mb-2">Training Visualization</h3>
                <TrainingVisualization />
              </div>
            )}
          </div>
        )}
        
        {/* Metrics Tab */}
        {activeTab === 'metrics' && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Model Metrics</h2>
            
            <div className="mb-4">
              <label className="block text-gray-700 mb-2">Select Model:</label>
              <select
                className="w-full p-2 border rounded-md"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="">Select a model</option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
            
            {selectedModel && <ModelMetrics modelId={selectedModel} />}
          </div>
        )}
      </main>
      
      <footer className="py-4 text-center text-gray-600">
        <p>Flappy Bird Reinforcement Learning Project</p>
      </footer>
    </div>
  );
} 