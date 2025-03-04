import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = '/api';

const MLflowExperiments = () => {
  const [experiments, setExperiments] = useState([]);
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Fetch experiments on component mount
  useEffect(() => {
    fetchExperiments();
  }, []);
  
  // Fetch runs when an experiment is selected
  useEffect(() => {
    if (selectedExperiment) {
      fetchRuns(selectedExperiment.experiment_id);
    }
  }, [selectedExperiment]);
  
  const fetchExperiments = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/mlflow/experiments`);
      if (response.data.status === 'success') {
        setExperiments(response.data.experiments);
        
        // Select the first experiment by default if available
        if (response.data.experiments.length > 0) {
          setSelectedExperiment(response.data.experiments[0]);
        }
      } else {
        setError('Failed to fetch experiments');
      }
    } catch (err) {
      console.error('Error fetching MLflow experiments:', err);
      setError('Failed to connect to MLflow. Is the server running?');
    } finally {
      setLoading(false);
    }
  };
  
  const fetchRuns = async (experimentId) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get(`${API_BASE_URL}/mlflow/runs?experiment_id=${experimentId}`);
      if (response.data.status === 'success') {
        // Sort runs by start time (newest first)
        const sortedRuns = response.data.runs.sort((a, b) => 
          b.start_time - a.start_time
        );
        setRuns(sortedRuns);
      } else {
        setError('Failed to fetch runs');
      }
    } catch (err) {
      console.error('Error fetching MLflow runs:', err);
      setError('Failed to fetch runs data');
    } finally {
      setLoading(false);
    }
  };
  
  // Format timestamp to readable date
  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleString();
  };
  
  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">MLflow Experiments</h2>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        {loading && <p className="text-gray-500">Loading...</p>}
        
        {experiments.length > 0 ? (
          <div>
            <div className="mb-4">
              <label htmlFor="experiment-select" className="block text-sm font-medium text-gray-700 mb-1">
                Select Experiment:
              </label>
              <select
                id="experiment-select"
                className="block w-full border-gray-300 rounded-md shadow-sm p-2 border"
                value={selectedExperiment?.experiment_id || ''}
                onChange={(e) => {
                  const selected = experiments.find(exp => exp.experiment_id === e.target.value);
                  setSelectedExperiment(selected);
                }}
              >
                {experiments.map((exp) => (
                  <option key={exp.experiment_id} value={exp.experiment_id}>
                    {exp.name}
                  </option>
                ))}
              </select>
            </div>
            
            {runs.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Run ID</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Start Time</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">End Time</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Best Score</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {runs.map((run) => (
                      <tr key={run.run_id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{run.run_id.slice(0, 8)}...</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{run.status}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatTimestamp(run.start_time)}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{run.end_time ? formatTimestamp(run.end_time) : 'In Progress'}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {run.metrics && run.metrics.best_score 
                            ? run.metrics.best_score.toFixed(2) 
                            : run.metrics && run.metrics.avg_score 
                              ? run.metrics.avg_score.toFixed(2)
                              : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-gray-500">No runs found for this experiment</p>
            )}
          </div>
        ) : (
          <p className="text-gray-500">No experiments found</p>
        )}
        
        <div className="mt-4">
          <p className="text-sm text-gray-500">
            Note: Access the full MLflow UI at <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">http://localhost:5000</a>
          </p>
          <p className="text-sm text-gray-500">
            Weights & Biases dashboard is available at <a href="https://wandb.ai" target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">https://wandb.ai</a>
          </p>
        </div>
      </div>
    </div>
  );
};

export default MLflowExperiments; 