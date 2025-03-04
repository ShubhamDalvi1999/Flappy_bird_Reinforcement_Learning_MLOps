import { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
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

const TrainingVisualization = () => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  const [trainingStats, setTrainingStats] = useState({
    episode: 0,
    totalEpisodes: 0,
    score: 0,
    reward: 0,
    bestScore: 0,
    epsilon: 1.0,
  });
  const [trainingHistory, setTrainingHistory] = useState({
    episodes: [],
    scores: [],
    rewards: [],
  });
  const canvasRef = useRef(null);

  // Connect to WebSocket server
  useEffect(() => {
    // Use relative URL for WebSocket connection
    const newSocket = io(window.location.origin, {
      path: '/socket.io',
      transports: ['websocket', 'polling'],
    });

    newSocket.on('connect', () => {
      console.log('Connected to WebSocket server');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
      setIsConnected(false);
    });

    newSocket.on('game_frame', (data) => {
      setCurrentFrame(`data:image/png;base64,${data.frame}`);
      setTrainingStats(prev => ({
        ...prev,
        score: data.score,
        episode: data.episode,
        epsilon: data.epsilon,
      }));
    });

    newSocket.on('training_step', (data) => {
      setTrainingStats(prev => ({
        ...prev,
        episode: data.episode,
        step: data.step,
        score: data.score,
        reward: data.reward,
      }));
    });

    newSocket.on('training_progress', (data) => {
      setTrainingStats({
        episode: data.episode,
        totalEpisodes: data.total_episodes,
        score: data.score,
        reward: data.reward,
        bestScore: data.best_score,
        epsilon: data.epsilon,
      });

      // Update training history
      setTrainingHistory(prev => {
        const newEpisodes = [...prev.episodes];
        const newScores = [...prev.scores];
        const newRewards = [...prev.rewards];

        // Only add if this episode isn't already in the history
        if (!newEpisodes.includes(data.episode)) {
          newEpisodes.push(data.episode);
          newScores.push(data.score);
          newRewards.push(data.reward);
        }

        return {
          episodes: newEpisodes,
          scores: newScores,
          rewards: newRewards,
        };
      });
    });

    newSocket.on('training_complete', (data) => {
      console.log('Training complete:', data);
    });

    newSocket.on('training_error', (data) => {
      console.error('Training error:', data);
    });

    setSocket(newSocket);

    return () => {
      newSocket.disconnect();
    };
  }, []);

  // Prepare chart data
  const chartData = {
    labels: trainingHistory.episodes,
    datasets: [
      {
        label: 'Score',
        data: trainingHistory.scores,
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
      },
      {
        label: 'Reward',
        data: trainingHistory.rewards,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      },
    ],
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
      },
    },
  };

  return (
    <div className="training-visualization">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">Training Visualization</h3>
          
          {isConnected ? (
            <div className="flex items-center text-green-600 mb-4">
              <div className="w-3 h-3 bg-green-600 rounded-full mr-2"></div>
              <span>Connected to training server</span>
            </div>
          ) : (
            <div className="flex items-center text-red-600 mb-4">
              <div className="w-3 h-3 bg-red-600 rounded-full mr-2"></div>
              <span>Disconnected from training server</span>
            </div>
          )}
          
          <div className="game-frame-container bg-gray-100 rounded-lg overflow-hidden mb-4" style={{ height: '300px' }}>
            {currentFrame ? (
              <img 
                src={currentFrame} 
                alt="Game Frame" 
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="flex items-center justify-center h-full text-gray-500">
                Waiting for training to start...
              </div>
            )}
          </div>
          
          <div className="training-stats grid grid-cols-2 gap-2">
            <div className="stat-box bg-blue-50 p-2 rounded">
              <div className="text-sm text-blue-700">Episode</div>
              <div className="font-bold">{trainingStats.episode} / {trainingStats.totalEpisodes}</div>
            </div>
            <div className="stat-box bg-green-50 p-2 rounded">
              <div className="text-sm text-green-700">Current Score</div>
              <div className="font-bold">{trainingStats.score}</div>
            </div>
            <div className="stat-box bg-purple-50 p-2 rounded">
              <div className="text-sm text-purple-700">Best Score</div>
              <div className="font-bold">{trainingStats.bestScore}</div>
            </div>
            <div className="stat-box bg-yellow-50 p-2 rounded">
              <div className="text-sm text-yellow-700">Epsilon</div>
              <div className="font-bold">{trainingStats.epsilon?.toFixed(4) || 0}</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">Training Progress</h3>
          {trainingHistory.episodes.length > 0 ? (
            <Line data={chartData} options={chartOptions} />
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-500">
              No training data available yet
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingVisualization; 