import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = '/api';

const WandBDashboard = () => {
  const [wandbStatus, setWandbStatus] = useState('loading');
  const [wandbData, setWandbData] = useState(null);
  const [gpuInfo, setGpuInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    fetchWandBStatus();
    fetchGpuInfo();
  }, []);
  
  const fetchWandBStatus = async () => {
    setLoading(true);
    try {
      // This endpoint would need to be implemented in the backend
      const response = await axios.get(`${API_BASE_URL}/wandb/status`);
      
      if (response.data.status === 'success') {
        setWandbStatus(response.data.wandb_status);
        if (response.data.recent_runs) {
          setWandbData(response.data.recent_runs);
        }
      } else {
        setWandbStatus('error');
        setError(response.data.message || 'Failed to get W&B status');
      }
    } catch (err) {
      console.error('Error fetching W&B status:', err);
      setWandbStatus('error');
      setError('Failed to connect to W&B API. Check your configuration and API key.');
    } finally {
      setLoading(false);
    }
  };
  
  const fetchGpuInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/gpu_check`);
      if (response.data.status === 'success') {
        setGpuInfo(response.data);
      }
    } catch (err) {
      console.error('Error fetching GPU info:', err);
    }
  };
  
  const renderGpuInfo = () => {
    if (!gpuInfo) return null;
    
    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h3 className="text-md font-medium text-gray-900 mb-2">GPU Information</h3>
        <div className="flex items-center mb-2">
          <div className={`w-3 h-3 rounded-full mr-2 ${gpuInfo.gpu_available ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-sm">
            {gpuInfo.gpu_available ? 
              `${gpuInfo.gpu_count} GPU(s) available` : 
              'No GPUs detected'}
          </span>
        </div>
        {gpuInfo.gpu_available && gpuInfo.gpu_info && (
          <div className="text-xs text-gray-600">
            <p>TensorFlow version: {gpuInfo.tf_version}</p>
            <div className="mt-1">
              {gpuInfo.gpu_info.map((gpu, idx) => (
                <p key={idx}>{gpu.name || `GPU ${idx}`}</p>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };
  
  if (loading) {
    return (
      <div className="bg-white overflow-hidden shadow rounded-lg">
        <div className="px-4 py-5 sm:p-6">
          <h2 className="text-lg font-medium text-gray-900 mb-4">Weights & Biases Dashboard</h2>
          <div className="flex justify-center">
            <p className="text-gray-500">Loading W&B information...</p>
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="bg-white overflow-hidden shadow rounded-lg">
      <div className="px-4 py-5 sm:p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Weights & Biases Dashboard</h2>
        
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4">
            <span className="block sm:inline">{error}</span>
          </div>
        )}
        
        <div className="mb-4">
          <div className="flex items-center mb-2">
            <div className={`w-3 h-3 rounded-full mr-2 ${wandbStatus === 'connected' ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
            <span className="font-medium">
              Status: {wandbStatus === 'connected' ? 'Connected to W&B' : 'Not connected'}
            </span>
          </div>
          
          {wandbStatus === 'connected' && wandbData ? (
            <div className="mt-4">
              <h3 className="text-md font-medium text-gray-900 mb-2">Recent Runs</h3>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Run Name</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Started</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Best Score</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {wandbData.map((run) => (
                      <tr key={run.id}>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{run.name}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{new Date(run.created_at).toLocaleString()}</td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          {run.summary && run.summary.best_score 
                            ? run.summary.best_score.toFixed(2) 
                            : 'N/A'}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                          <span className={`px-2 py-1 text-xs rounded-full ${
                            run.state === 'running' ? 'bg-blue-100 text-blue-800' :
                            run.state === 'finished' ? 'bg-green-100 text-green-800' :
                            'bg-gray-100 text-gray-800'
                          }`}>
                            {run.state}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="mt-4 p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                {wandbStatus === 'connected' 
                  ? 'No recent W&B runs found'
                  : 'Connect to Weights & Biases to see your runs and metrics'}
              </p>
              
              <div className="mt-3">
                <a 
                  href="https://wandb.ai" 
                  target="_blank" 
                  rel="noreferrer" 
                  className="text-blue-500 hover:underline text-sm"
                >
                  Open W&B Dashboard →
                </a>
              </div>
            </div>
          )}
        </div>
        
        {renderGpuInfo()}
        
        <div className="mt-6 border-t pt-4">
          <h3 className="text-md font-medium text-gray-900 mb-2">Setup Instructions</h3>
          <ol className="list-decimal list-inside text-sm text-gray-600 space-y-1">
            <li>Sign up for a free account at <a href="https://wandb.ai" target="_blank" rel="noreferrer" className="text-blue-500 hover:underline">wandb.ai</a></li>
            <li>Get your API key from your W&B settings page</li>
            <li>Set your API key in the environment: <code className="bg-gray-100 px-1 py-0.5 rounded">export WANDB_API_KEY=your_api_key</code></li>
            <li>Restart the application with W&B enabled</li>
          </ol>
        </div>
      </div>
    </div>
  );
};

export default WandBDashboard; 