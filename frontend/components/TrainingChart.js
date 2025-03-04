import { useEffect, useState } from 'react';
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
import { Line } from 'react-chartjs-2';

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

const TrainingChart = ({ progressData, chartType = 'scores' }) => {
  const [chartData, setChartData] = useState(null);
  const [options, setOptions] = useState(null);
  
  useEffect(() => {
    if (!progressData) return;
    
    // Determine which data series to show based on chartType
    let labels = [];
    let dataPoints = [];
    let label = '';
    let borderColor = '';
    let backgroundColor = '';
    
    switch (chartType) {
      case 'scores':
        labels = progressData.episodes || [];
        dataPoints = progressData.scores || [];
        label = 'Episode Scores';
        borderColor = 'rgb(53, 162, 235)';
        backgroundColor = 'rgba(53, 162, 235, 0.5)';
        break;
      case 'avg_scores':
        labels = progressData.episodes || [];
        dataPoints = progressData.avg_scores || [];
        label = 'Average Scores (last 100 episodes)';
        borderColor = 'rgb(75, 192, 192)';
        backgroundColor = 'rgba(75, 192, 192, 0.5)';
        break;
      case 'epsilons':
        labels = progressData.episodes || [];
        dataPoints = progressData.epsilons || [];
        label = 'Exploration Rate (Epsilon)';
        borderColor = 'rgb(255, 99, 132)';
        backgroundColor = 'rgba(255, 99, 132, 0.5)';
        break;
      case 'losses':
        labels = progressData.episodes || [];
        dataPoints = progressData.losses || [];
        label = 'Loss';
        borderColor = 'rgb(255, 159, 64)';
        backgroundColor = 'rgba(255, 159, 64, 0.5)';
        break;
      default:
        labels = progressData.episodes || [];
        dataPoints = progressData.scores || [];
        label = 'Episode Scores';
        borderColor = 'rgb(53, 162, 235)';
        backgroundColor = 'rgba(53, 162, 235, 0.5)';
    }
    
    // Create chart data
    setChartData({
      labels,
      datasets: [
        {
          label,
          data: dataPoints,
          borderColor,
          backgroundColor,
          tension: 0.1,
        },
      ],
    });
    
    // Set chart options
    setOptions({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: `Training Progress - ${label}`,
        },
        tooltip: {
          mode: 'index',
          intersect: false,
        },
      },
      scales: {
        y: {
          beginAtZero: chartType === 'epsilons',
        },
      },
      animation: {
        duration: 0, // Disable animation for better performance with large datasets
      },
    });
  }, [progressData, chartType]);
  
  if (!chartData || !options) {
    return (
      <div className="flex justify-center items-center h-64 bg-white rounded-lg shadow">
        <p className="text-gray-500">No training data available</p>
      </div>
    );
  }
  
  return (
    <div className="h-64 bg-white rounded-lg shadow p-4">
      <Line data={chartData} options={options} />
    </div>
  );
};

export default TrainingChart; 