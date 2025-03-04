import { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const API_BASE_URL = '/api';

const ModelMetrics = () => {
  const [selectedModel, setSelectedModel] = useState('');
  const [models, setModels] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch available models on component mount
  useEffect(() => {
    fetchModels();
  }, []);

  // Fetch model metrics when a model is selected
  useEffect(() => {
    if (selectedModel) {
      fetchModelMetrics(selectedModel);
      fetchTrainingHistory(selectedModel);
    }
  }, [selectedModel]);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/models`);
      if (response.data.status === 'success') {
        setModels(response.data.models);
        
        // Select the first model by default if available
        if (response.data.models.length > 0) {
          setSelectedModel(response.data.models[0].id);
        }
      } else {
        setError('Failed to fetch models');
      }
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to connect to the API. Is the server running?');
    } finally {
      setLoading(false);
    }
  };

  const fetchModelMetrics = async (modelId) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/models/${modelId}/metrics`);
      if (response.data.status === 'success') {
        setMetrics(response.data.metrics);
      } else {
        setError('Failed to fetch model metrics');
      }
    } catch (err) {
      console.error('Error fetching model metrics:', err);
      setError('Failed to fetch model metrics');
    } finally {
      setLoading(false);
    }
  };

  const fetchTrainingHistory = async (modelId) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/models/${modelId}/history`);
      if (response.data.status === 'success') {
        setTrainingHistory(response.data.history);
      } else {
        setError('Failed to fetch training history');
      }
    } catch (err) {
      console.error('Error fetching training history:', err);
      setError('Failed to fetch training history');
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data from training history
  const prepareChartData = () => {
    if (!trainingHistory || !trainingHistory.episodes || !trainingHistory.scores) {
      return null;
    }

    // For rolling average line
    const windowSize = 100;
    const rollingAvg = [];
    
    trainingHistory.scores.forEach((_, index) => {
      if (index >= windowSize - 1) {
        const slice = trainingHistory.scores.slice(index - windowSize + 1, index + 1);
        const avg = slice.reduce((sum, val) => sum + val, 0) / windowSize;
        rollingAvg.push(avg);
      } else {
        rollingAvg.push(null);
      }
    });

    return {
      labels: trainingHistory.episodes,
      datasets: [
        {
          label: 'Episode Score',
          data: trainingHistory.scores,
          borderColor: 'rgba(75, 192, 192, 0.2)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          pointRadius: 0,
          borderWidth: 1,
        },
        {
          label: `${windowSize}-Episode Rolling Average`,
          data: rollingAvg,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.2)',
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Training Progress',
      },
    },
    scales: {
      y: {
        title: {
          display: true,
          text: 'Score',
        },
      },
      x: {
        title: {
          display: true,
          text: 'Episode',
        },
      },
    },
  };

  const chartData = prepareChartData();

  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Model Metrics</h2>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {loading && <p className="text-gray-500">Loading...</p>}
        
        <div className="mb-4">
          <label htmlFor="model-select" className="block text-sm font-medium text-gray-700 mb-1">
            Select Model:
          </label>
          <select
            id="model-select"
            className="block w-full border-gray-300 rounded-md shadow-sm p-2 border"
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
        
        {metrics && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-blue-900">Max Score</h3>
              <p className="mt-1 text-2xl font-semibold text-blue-900">{metrics.max_score}</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-green-900">Average Score</h3>
              <p className="mt-1 text-2xl font-semibold text-green-900">{metrics.avg_score.toFixed(2)}</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <h3 className="text-sm font-medium text-purple-900">Training Episodes</h3>
              <p className="mt-1 text-2xl font-semibold text-purple-900">{metrics.episodes_trained}</p>
            </div>
          </div>
        )}
        
        {chartData && (
          <div className="mt-6">
            <h3 className="text-md font-medium text-gray-900 mb-2">Training Progress</h3>
            <div className="h-80">
              <Line data={chartData} options={chartOptions} />
            </div>
          </div>
        )}
        
        {metrics && (
          <div className="mt-6">
            <h3 className="text-md font-medium text-gray-900 mb-2">Additional Information</h3>
            <div className="bg-gray-50 p-4 rounded-lg">
              <dl className="grid grid-cols-1 gap-x-4 gap-y-2 sm:grid-cols-2">
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Learning Rate</dt>
                  <dd className="mt-1 text-sm text-gray-900">{metrics.learning_rate}</dd>
                </div>
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Discount Factor</dt>
                  <dd className="mt-1 text-sm text-gray-900">{metrics.discount_factor}</dd>
                </div>
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Training Time</dt>
                  <dd className="mt-1 text-sm text-gray-900">{metrics.training_time} minutes</dd>
                </div>
                <div className="sm:col-span-1">
                  <dt className="text-sm font-medium text-gray-500">Model Architecture</dt>
                  <dd className="mt-1 text-sm text-gray-900">{metrics.architecture}</dd>
                </div>
              </dl>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelMetrics; 