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

const ModelMetrics = ({ modelId }) => {
  const [metrics, setMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fetch model metrics when modelId changes
  useEffect(() => {
    if (modelId) {
      fetchModelMetrics();
      fetchTrainingHistory();
    }
  }, [modelId]);

  const fetchModelMetrics = async () => {
    setLoading(true);
    try {
      // In a real app, this would fetch metrics from the backend
      // Since we removed MLflow, we'll use mock data
      setMetrics({
        max_score: 42,
        avg_score: 15.7,
        episodes_trained: 1000,
        learning_rate: 0.001,
        discount_factor: 0.99,
        training_time: 30,
        architecture: 'DQN',
        epsilon_final: 0.01,
        total_steps: 50000,
        best_episode_reward: 100
      });
      setError(null);
    } catch (error) {
      console.error('Error fetching model metrics:', error);
      setError('Failed to fetch model metrics');
    } finally {
      setLoading(false);
    }
  };

  const fetchTrainingHistory = async () => {
    setLoading(true);
    try {
      // In a real app, this would fetch training history from the backend
      // Since we removed MLflow, we'll use mock data
      const episodes = Array.from({ length: 100 }, (_, i) => i * 10);
      const scores = episodes.map(e => Math.min(100, Math.max(0, 50 + e/10 + Math.random() * 30 - 15)));
      
      setTrainingHistory({
        episodes,
        scores
      });
      setError(null);
    } catch (error) {
      console.error('Error fetching training history:', error);
      setError('Failed to fetch training history');
    } finally {
      setLoading(false);
    }
  };

  const prepareChartData = () => {
    if (!trainingHistory) return null;

    return {
      labels: trainingHistory.episodes,
      datasets: [
        {
          label: 'Score',
          data: trainingHistory.scores,
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.5)',
          tension: 0.1
        }
      ]
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
        beginAtZero: true,
        title: {
          display: true,
          text: 'Score'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Episode'
        }
      }
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
        <strong className="font-bold">Error!</strong>
        <span className="block sm:inline"> {error}</span>
      </div>
    );
  }

  if (!metrics || !trainingHistory) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-500">Select a model to view metrics</p>
      </div>
    );
  }

  const chartData = prepareChartData();

  return (
    <div className="space-y-6">
      <div className="bg-white shadow overflow-hidden sm:rounded-lg">
        <div className="px-4 py-5 sm:px-6">
          <h3 className="text-lg leading-6 font-medium text-gray-900">Model Performance Metrics</h3>
          <p className="mt-1 max-w-2xl text-sm text-gray-500">Key metrics for the selected model</p>
        </div>
        <div className="border-t border-gray-200">
          <dl>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Maximum Score</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{metrics.max_score}</dd>
            </div>
            <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Average Score</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{metrics.avg_score.toFixed(2)}</dd>
            </div>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Episodes Trained</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{metrics.episodes_trained}</dd>
            </div>
            <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Learning Rate</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{metrics.learning_rate}</dd>
            </div>
            <div className="bg-gray-50 px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Discount Factor (Gamma)</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{metrics.discount_factor}</dd>
            </div>
            <div className="bg-white px-4 py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
              <dt className="text-sm font-medium text-gray-500">Training Time (minutes)</dt>
              <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">{metrics.training_time}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
        <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">Training Progress</h3>
        {chartData && <Line data={chartData} options={chartOptions} />}
      </div>
    </div>
  );
};

export default ModelMetrics; 