import { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const API_BASE_URL = '/api';
const CANVAS_WIDTH = 400;
const CANVAS_HEIGHT = 600;
const BIRD_WIDTH = 34;
const BIRD_HEIGHT = 24;
const PIPE_WIDTH = 52;
const GAME_SPEED = 30; // frames per second

const GameComponent = ({ selectedModel }) => {
  const canvasRef = useRef(null);
  const [gameState, setGameState] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gameScore, setGameScore] = useState(0);
  const [gameOver, setGameOver] = useState(false);
  const [highScore, setHighScore] = useState(0);
  const [birdImg, setBirdImg] = useState(null);
  const [pipeImg, setPipeImg] = useState(null);
  const [bgImg, setBgImg] = useState(null);
  const animationRef = useRef(null);
  
  // Load images on component mount
  useEffect(() => {
    // Load bird image
    const bird = new Image();
    bird.src = '/images/bird.png';
    bird.onload = () => setBirdImg(bird);
    
    // Load pipe image
    const pipe = new Image();
    pipe.src = '/images/pipe.png';
    pipe.onload = () => setPipeImg(pipe);
    
    // Load background image
    const bg = new Image();
    bg.src = '/images/background.png';
    bg.onload = () => setBgImg(bg);
    
    return () => {
      // Clean up any game loops on unmount
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);
  
  // Reset game when model changes
  useEffect(() => {
    if (isPlaying) {
      stopGame();
    }
    setGameOver(false);
    setGameScore(0);
  }, [selectedModel]);
  
  const startGame = async () => {
    if (!selectedModel) {
      alert('Please select a model first');
      return;
    }
    
    try {
      const response = await axios.post(`${API_BASE_URL}/game/start`, {
        model_id: selectedModel
      });
      
      if (response.data.status === 'success') {
        setGameState(response.data.initial_state);
        setIsPlaying(true);
        setGameOver(false);
        setGameScore(0);
        
        // Start the game loop
        startGameLoop();
      }
    } catch (error) {
      console.error('Error starting game:', error);
      alert('Failed to start game. Is the server running?');
    }
  };
  
  const stopGame = async () => {
    try {
      await axios.post(`${API_BASE_URL}/game/stop`);
      setIsPlaying(false);
      
      // Stop the game loop
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    } catch (error) {
      console.error('Error stopping game:', error);
    }
  };
  
  const getNextFrame = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/game/step`);
      
      if (response.data.status === 'success') {
        setGameState(response.data.state);
        setGameScore(response.data.state.score);
        
        if (response.data.state.done) {
          setGameOver(true);
          setIsPlaying(false);
          
          // Update high score if needed
          if (response.data.state.score > highScore) {
            setHighScore(response.data.state.score);
          }
          
          if (animationRef.current) {
            cancelAnimationFrame(animationRef.current);
            animationRef.current = null;
          }
        }
      }
    } catch (error) {
      console.error('Error getting next frame:', error);
      setIsPlaying(false);
    }
  };
  
  const startGameLoop = () => {
    let lastTime = 0;
    const frameInterval = 1000 / GAME_SPEED;
    
    const gameLoop = async (timestamp) => {
      const deltaTime = timestamp - lastTime;
      
      if (deltaTime >= frameInterval) {
        await getNextFrame();
        lastTime = timestamp;
      }
      
      // Continue the loop if still playing
      if (isPlaying) {
        animationRef.current = requestAnimationFrame(gameLoop);
      }
    };
    
    animationRef.current = requestAnimationFrame(gameLoop);
  };
  
  // Draw the game on canvas
  useEffect(() => {
    if (!canvasRef.current || !gameState || !birdImg || !pipeImg || !bgImg) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Draw background
    ctx.drawImage(bgImg, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Draw pipes
    gameState.pipes.forEach(pipe => {
      // Draw top pipe (flipped)
      ctx.save();
      ctx.translate(pipe.x, pipe.y - gameState.pipe_gap / 2);
      ctx.scale(1, -1);
      ctx.drawImage(pipeImg, -PIPE_WIDTH / 2, 0, PIPE_WIDTH, 320);
      ctx.restore();
      
      // Draw bottom pipe
      ctx.drawImage(pipeImg, pipe.x - PIPE_WIDTH / 2, pipe.y + gameState.pipe_gap / 2, PIPE_WIDTH, 320);
    });
    
    // Draw bird
    ctx.save();
    ctx.translate(gameState.bird_x, gameState.bird_y);
    
    // Rotate bird based on velocity
    const rotation = Math.min(Math.max(gameState.bird_velocity * 0.1, -0.5), 0.5);
    ctx.rotate(rotation);
    
    ctx.drawImage(birdImg, -BIRD_WIDTH / 2, -BIRD_HEIGHT / 2, BIRD_WIDTH, BIRD_HEIGHT);
    ctx.restore();
    
    // Draw score
    ctx.fillStyle = 'white';
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 3;
    ctx.font = 'bold 36px Arial';
    ctx.textAlign = 'center';
    ctx.strokeText(gameState.score, CANVAS_WIDTH / 2, 50);
    ctx.fillText(gameState.score, CANVAS_WIDTH / 2, 50);
    
  }, [gameState, birdImg, pipeImg, bgImg]);
  
  return (
    <div className="game-container">
      <div className="canvas-wrapper relative">
        <canvas 
          ref={canvasRef} 
          width={CANVAS_WIDTH} 
          height={CANVAS_HEIGHT}
          className="border border-gray-300 bg-blue-50"
        />
        
        {!isPlaying && !gameOver && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-40 text-white">
            <h3 className="text-2xl font-bold mb-4">Flappy Bird RL</h3>
            <button 
              onClick={startGame}
              className="px-4 py-2 bg-green-600 rounded-lg hover:bg-green-700 transition-colors"
            >
              Start Game
            </button>
          </div>
        )}
        
        {gameOver && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black bg-opacity-40 text-white">
            <h3 className="text-2xl font-bold mb-2">Game Over</h3>
            <p className="text-xl mb-4">Score: {gameScore}</p>
            <button 
              onClick={startGame}
              className="px-4 py-2 bg-green-600 rounded-lg hover:bg-green-700 transition-colors"
            >
              Play Again
            </button>
          </div>
        )}
      </div>
      
      <div className="stats-container mt-4 grid grid-cols-2 gap-4">
        <div className="bg-blue-50 p-2 rounded-lg text-center">
          <p className="text-sm text-blue-700 font-medium">Current Score</p>
          <p className="text-xl font-bold">{gameScore}</p>
        </div>
        <div className="bg-green-50 p-2 rounded-lg text-center">
          <p className="text-sm text-green-700 font-medium">High Score</p>
          <p className="text-xl font-bold">{highScore}</p>
        </div>
      </div>
      
      {isPlaying && (
        <div className="mt-4 text-center">
          <button
            onClick={stopGame}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Stop Game
          </button>
        </div>
      )}
      
      <div className="mt-4">
        <h3 className="text-sm font-medium text-gray-700 mb-2">Game Controls</h3>
        <p className="text-sm text-gray-600">
          This demonstration is fully autonomous. The AI agent is playing the game based on the model you selected.
        </p>
      </div>
    </div>
  );
};

export default GameComponent; 