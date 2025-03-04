import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlappyBirdEnv(gym.Env):
    """Custom Flappy Bird environment that follows gym interface"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render=False):
        super(FlappyBirdEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 0 (do nothing), 1 (flap)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: bird y position, vertical velocity, distance to next pipe, height of next pipe
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Game parameters
        self.window_width = 400
        self.window_height = 600
        self.pipe_width = 70
        self.pipe_gap = 150
        self.bird_width = 30
        self.bird_height = 30
        self.gravity = 1
        self.flap_strength = -10
        self.pipe_velocity = -4
        self.max_velocity = 10
        self.min_pipe_y = 100
        self.max_score = 100  # Terminal condition
        
        # Initialize game state
        self.bird_x = self.window_width // 3
        self.bird_y = self.window_height // 2
        self.bird_velocity = 0
        self.pipes = [
            {'x': self.window_width + 100, 
             'y': np.random.randint(self.min_pipe_y, self.window_height - self.pipe_gap - self.min_pipe_y)}
        ]
        self.score = 0
        self.steps = 0
        self.num_pipes = 3
        self.render_mode = 'human' if render else None
        
        # For compatibility with older gym versions
        self.state_size = self.observation_space.shape[0]
        self.action_size = self.action_space.n
        
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Reset game state
        self.bird_y = self.window_height // 2
        self.bird_velocity = 0
        self.pipes = [
            {'x': self.window_width + 100, 
             'y': np.random.randint(self.min_pipe_y, self.window_height - self.pipe_gap - self.min_pipe_y)}
        ]
        self.score = 0
        self.steps = 0
        
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        self.steps += 1
        
        # Apply action (flap)
        if action == 1:
            self.bird_velocity = self.flap_strength
        
        # Apply gravity and update bird position
        self.bird_velocity = min(self.bird_velocity + self.gravity, self.max_velocity)
        self.bird_y += self.bird_velocity
        
        # Update pipes
        self._update_pipes()
        
        # Check for collisions
        done = self._check_collision()
        
        # Check if reached max score (optional terminal condition)
        if self.score >= self.max_score:
            done = True
        
        # Calculate reward
        if done:
            reward = -10  # Penalty for collision
        else:
            reward = 0.1  # Small reward for surviving
            
            # Additional reward for passing a pipe
            for pipe in self.pipes:
                if self.bird_x > pipe['x'] and self.bird_x <= pipe['x'] + self.pipe_velocity:
                    reward += 1.0
        
        state = self._get_state()
        info = {'score': self.score}
        
        return state, reward, done, False, info
    
    def _check_collision(self):
        # Check if bird hits the ground or ceiling
        if self.bird_y <= 0 or self.bird_y >= self.window_height:
            return True
        
        # Check for collision with pipes
        for pipe in self.pipes:
            # Top pipe
            if (self.bird_x + self.bird_width > pipe['x'] and 
                self.bird_x < pipe['x'] + self.pipe_width and 
                self.bird_y < pipe['y']):
                return True
            
            # Bottom pipe
            if (self.bird_x + self.bird_width > pipe['x'] and 
                self.bird_x < pipe['x'] + self.pipe_width and 
                self.bird_y + self.bird_height > pipe['y'] + self.pipe_gap):
                return True
                
        return False
    
    def _update_pipes(self):
        # Move pipes
        for pipe in self.pipes:
            pipe['x'] += self.pipe_velocity
        
        # Remove pipes that are off-screen
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + self.pipe_width > 0]
        
        # Add new pipes if needed
        if len(self.pipes) < self.num_pipes:
            last_pipe = max(self.pipes, key=lambda p: p['x']) if self.pipes else {'x': 0}
            new_pipe_x = max(last_pipe['x'] + self.window_width // 2, self.window_width)
            new_pipe_y = np.random.randint(self.min_pipe_y, self.window_height - self.pipe_gap - self.min_pipe_y)
            self.pipes.append({'x': new_pipe_x, 'y': new_pipe_y})
        
        # Update score
        for pipe in self.pipes:
            if self.bird_x > pipe['x'] + self.pipe_width and not hasattr(pipe, 'passed'):
                pipe['passed'] = True
                self.score += 1
    
    def _get_state(self):
        # Find the nearest pipe
        nearest_pipe = None
        for pipe in self.pipes:
            if pipe['x'] + self.pipe_width > self.bird_x:
                if nearest_pipe is None or pipe['x'] < nearest_pipe['x']:
                    nearest_pipe = pipe
        
        if nearest_pipe is None:
            # If no pipe ahead, use a default value
            nearest_pipe_dist = self.window_width
            nearest_pipe_y = self.window_height // 2
        else:
            nearest_pipe_dist = nearest_pipe['x'] - self.bird_x
            nearest_pipe_y = nearest_pipe['y']
        
        # Normalize observations for better learning
        state = np.array([
            self.bird_y / self.window_height,  # Bird y-position (normalized)
            self.bird_velocity / self.max_velocity,  # Bird velocity (normalized)
            nearest_pipe_dist / self.window_width,  # Horizontal distance to pipe (normalized)
            nearest_pipe_y / self.window_height  # Height of pipe gap (normalized)
        ], dtype=np.float32)
        
        return state
    
    def render(self):
        # This is a simplified render method since we're not using pygame in the Docker container
        if self.render_mode == 'human':
            print(f"Bird position: ({self.bird_x}, {self.bird_y}), Velocity: {self.bird_velocity}, Score: {self.score}")
            
    def close(self):
        pass

# For compatibility with gym.make
def make_env():
    return FlappyBirdEnv()

# Export environment for use in other files
env = FlappyBirdEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 