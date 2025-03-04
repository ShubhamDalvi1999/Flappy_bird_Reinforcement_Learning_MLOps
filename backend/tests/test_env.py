import sys
import os
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBirdEnv

class TestFlappyBirdEnv:
    
    @pytest.fixture
    def env(self):
        """Create a test environment"""
        return FlappyBirdEnv(render=False)
    
    def test_env_initialization(self, env):
        """Test environment initialization"""
        assert env.action_size == 2  # Flap or not flap
        assert env.state_size > 0
        assert env.window_width > 0
        assert env.window_height > 0
        assert env.pipe_gap > 0
        
        # Initial bird position
        assert 0 <= env.bird_x <= env.window_width
        assert 0 <= env.bird_y <= env.window_height
        
        # Initial pipes
        assert len(env.pipes) > 0
        for pipe in env.pipes:
            assert 'x' in pipe
            assert 'y' in pipe
    
    def test_reset(self, env):
        """Test environment reset"""
        initial_state = env.reset()
        
        # State should be a numpy array with the correct dimension
        assert isinstance(initial_state, np.ndarray)
        assert initial_state.shape == (env.state_size,)
        
        # Bird should be at the initial position
        assert env.bird_x == env.window_width // 3
        assert env.bird_y == env.window_height // 2
        
        # Bird velocity should be 0
        assert env.bird_velocity == 0
        
        # Score should be 0
        assert env.score == 0
    
    def test_step_no_flap(self, env):
        """Test step method with no flap action"""
        env.reset()
        initial_y = env.bird_y
        initial_velocity = env.bird_velocity
        
        next_state, reward, done, info = env.step(0)  # No flap
        
        # Bird should fall (y increases in pygame coordinates)
        assert env.bird_y > initial_y
        
        # Velocity should increase (downward)
        assert env.bird_velocity > initial_velocity
        
        # State should be updated
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (env.state_size,)
        
        # Reward should be returned
        assert isinstance(reward, float)
        
        # Done flag should be returned
        assert isinstance(done, bool)
        
        # Info dict should be returned
        assert isinstance(info, dict)
    
    def test_step_flap(self, env):
        """Test step method with flap action"""
        env.reset()
        
        # Force bird to have downward velocity
        env.bird_velocity = 5
        initial_y = env.bird_y
        
        next_state, reward, done, info = env.step(1)  # Flap
        
        # Velocity should be negative (upward)
        assert env.bird_velocity < 0
        
        # State should be updated
        assert isinstance(next_state, np.ndarray)
        assert next_state.shape == (env.state_size,)
    
    def test_collision_detection(self, env):
        """Test collision detection"""
        env.reset()
        
        # No collision initially
        assert not env._check_collision()
        
        # Force bird to go below the screen
        env.bird_y = env.window_height + 10
        assert env._check_collision()
        
        # Reset and force bird to go above the screen
        env.reset()
        env.bird_y = -10
        assert env._check_collision()
        
        # Reset and force collision with pipe
        env.reset()
        # Position bird inside a pipe
        env.bird_x = env.pipes[0]['x']
        env.bird_y = env.pipes[0]['y'] - env.pipe_gap // 2 - 10  # Above the gap
        assert env._check_collision()
    
    def test_passing_pipes(self, env):
        """Test passing pipes and score increment"""
        env.reset()
        
        # Get pipes
        pipes = env.pipes
        
        # Position bird just before passing a pipe
        env.bird_x = pipes[0]['x'] - 5
        env.bird_y = pipes[0]['y']  # Inside the gap
        
        # Take a step to pass the pipe
        _, reward, _, _ = env.step(0)
        
        # Bird should be past the pipe now
        assert env.bird_x > pipes[0]['x']
        
        # Score should increment
        assert env.score > 0
        
        # Reward should be positive
        assert reward > 0
    
    def test_multiple_steps(self, env):
        """Test multiple steps"""
        env.reset()
        
        # Take 100 steps or until done
        for _ in range(100):
            # Randomly choose action
            action = np.random.randint(0, 2)
            
            _, _, done, _ = env.step(action)
            if done:
                break
        
        # Environment should handle multiple steps without errors
        assert True
    
    def test_get_state(self, env):
        """Test state generation"""
        env.reset()
        
        state = env._get_state()
        
        # State should have the correct shape
        assert isinstance(state, np.ndarray)
        assert state.shape == (env.state_size,)
        
        # State should include bird information
        assert env.bird_y in state or env.bird_y/env.window_height in state
        
        # State should include closest pipe information
        assert any(pipe['x'] in state or pipe['x']/env.window_width in state for pipe in env.pipes)
    
    def test_pipe_generation(self, env):
        """Test pipe generation"""
        env.reset()
        
        # Store initial pipes
        initial_pipes = env.pipes.copy()
        
        # Move bird past all pipes
        for _ in range(len(env.pipes) * 2):
            env.bird_x += env.window_width // 4
            env._update_pipes()
        
        # Should have generated new pipes
        assert env.pipes != initial_pipes
        
        # Should always maintain the correct number of pipes
        assert len(env.pipes) == env.num_pipes
        
        # All pipes should be within window width
        for pipe in env.pipes:
            assert 0 <= pipe['x'] <= env.window_width + 100  # Allow slight buffer
    
    def test_close(self, env):
        """Test environment close method"""
        # This test just ensures the close method doesn't error
        env.close()
        assert True


if __name__ == "__main__":
    pytest.main(["-v"]) 