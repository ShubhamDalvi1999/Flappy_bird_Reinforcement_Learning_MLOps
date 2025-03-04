import { useEffect, useRef, useState } from 'react';

const Game = ({ gameState }) => {
  const canvasRef = useRef(null);
  const [animationFrameId, setAnimationFrameId] = useState(null);
  
  // Game rendering constants
  const SCREEN_WIDTH = 400;
  const SCREEN_HEIGHT = 600;
  const BIRD_WIDTH = 30;
  const BIRD_HEIGHT = 30;
  const PIPE_WIDTH = 70;
  const BIRD_X_POS = SCREEN_WIDTH / 4;
  const SKY_COLOR = '#70c5ce';
  const BIRD_COLOR = '#ffcc00';
  const PIPE_COLOR = '#0bcc51';
  
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = SCREEN_WIDTH;
    canvas.height = SCREEN_HEIGHT;
    
    // Function to render the game state
    const render = () => {
      // Clear canvas
      ctx.fillStyle = SKY_COLOR;
      ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
      
      if (gameState) {
        // Draw pipes if we have game state
        if (gameState.pipes && gameState.pipes.length > 0) {
          gameState.pipes.forEach(pipe => {
            // Draw top pipe
            ctx.fillStyle = PIPE_COLOR;
            ctx.fillRect(pipe.x, 0, PIPE_WIDTH, pipe.y);
            
            // Draw bottom pipe
            ctx.fillRect(
              pipe.x,
              pipe.y + gameState.pipeGap,
              PIPE_WIDTH,
              SCREEN_HEIGHT - pipe.y - gameState.pipeGap
            );
          });
        }
        
        // Draw bird
        ctx.fillStyle = BIRD_COLOR;
        ctx.fillRect(BIRD_X_POS, gameState.birdY, BIRD_WIDTH, BIRD_HEIGHT);
        
        // Display score
        ctx.fillStyle = '#000';
        ctx.font = '24px Arial';
        ctx.fillText(`Score: ${gameState.score || 0}`, 10, 30);
      } else {
        // Draw a default bird if no game state
        ctx.fillStyle = BIRD_COLOR;
        ctx.fillRect(BIRD_X_POS, SCREEN_HEIGHT / 2, BIRD_WIDTH, BIRD_HEIGHT);
        
        // Draw placeholder text
        ctx.fillStyle = '#000';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Start training to see the game', SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 50);
      }
    };
    
    // Initial render
    render();
    
    // Set up animation loop if game state is available
    if (gameState) {
      const animate = () => {
        render();
        setAnimationFrameId(requestAnimationFrame(animate));
      };
      
      animate();
    }
    
    // Cleanup animation on unmount
    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [gameState]); // Re-render when game state changes
  
  return (
    <div className="game-container bg-flappy-blue">
      <canvas 
        ref={canvasRef} 
        className="mx-auto" 
        width={SCREEN_WIDTH} 
        height={SCREEN_HEIGHT}
      />
    </div>
  );
};

export default Game; 