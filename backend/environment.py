import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random

class FlappyBirdEnv(gym.Env):
    """Custom Flappy Bird environment that follows gym interface"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 0 (do nothing), 1 (flap)
        self.action_space = spaces.Discrete(2)
        
        # Observation space: bird y position, vertical velocity, distance to next pipe, height of next pipe
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        
        # Game parameters
        self.screen_width = 400
        self.screen_height = 600
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
        self.bird_y = None
        self.bird_velocity = None
        self.pipes = None
        self.score = None
        self.steps = None
        
        # Initialize rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.bird_y = self.screen_height // 2
        self.bird_velocity = 0
        self.pipes = [
            {'x': self.screen_width + 100, 
             'y': random.randint(self.min_pipe_y, self.screen_height - self.pipe_gap - self.min_pipe_y)}
        ]
        self.score = 0
        self.steps = 0
        
        # Initialize rendering if needed
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Flappy Bird RL")
        
        if self.render_mode == "human" and self.clock is None:
            self.clock = pygame.time.Clock()
            
        obs = self._get_observation()
        info = {}
        
        return obs, info
    
    def step(self, action):
        self.steps += 1
        
        # Apply action (flap)
        if action == 1:
            self.bird_velocity = self.flap_strength
        
        # Apply gravity and update bird position
        self.bird_velocity = min(self.bird_velocity + self.gravity, self.max_velocity)
        self.bird_y += self.bird_velocity
        
        # Update pipes
        for pipe in self.pipes:
            pipe['x'] += self.pipe_velocity
        
        # Remove pipes that are off-screen
        if self.pipes and self.pipes[0]['x'] < -self.pipe_width:
            self.pipes.pop(0)
            self.score += 1
        
        # Add new pipes
        if not self.pipes or self.pipes[-1]['x'] < self.screen_width - 200:
            pipe_y = random.randint(self.min_pipe_y, self.screen_height - self.pipe_gap - self.min_pipe_y)
            self.pipes.append({'x': self.screen_width, 'y': pipe_y})
        
        # Check for collisions
        terminated = self._check_collision()
        
        # Check if reached max score (optional terminal condition)
        if self.score >= self.max_score:
            terminated = True
        
        # Truncated if we've reached maximum steps
        truncated = False
        
        # Calculate reward
        if terminated:
            reward = -10  # Penalty for collision
        else:
            reward = 0.1  # Small reward for surviving
            
            # Additional reward for passing a pipe
            if any(pipe['x'] + self.pipe_width // 2 == self.screen_width // 4 for pipe in self.pipes):
                reward += 1.0
        
        obs = self._get_observation()
        info = {'score': self.score}
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info
    
    def _check_collision(self):
        # Check if bird hits the ground or ceiling
        if self.bird_y <= 0 or self.bird_y >= self.screen_height:
            return True
        
        # Check for collision with pipes
        bird_rect = pygame.Rect(self.screen_width // 4, self.bird_y, self.bird_width, self.bird_height)
        
        for pipe in self.pipes:
            # Top pipe rect
            top_pipe_rect = pygame.Rect(pipe['x'], 0, self.pipe_width, pipe['y'])
            
            # Bottom pipe rect
            bottom_pipe_rect = pygame.Rect(
                pipe['x'], 
                pipe['y'] + self.pipe_gap, 
                self.pipe_width, 
                self.screen_height - pipe['y'] - self.pipe_gap
            )
            
            # Check collision
            if bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect):
                return True
                
        return False
    
    def _get_observation(self):
        # Find the nearest pipe
        if not self.pipes:
            nearest_pipe_x = self.screen_width
            nearest_pipe_y = self.screen_height // 2
        else:
            # Get first pipe ahead of the bird
            nearest_pipe = None
            for pipe in self.pipes:
                if pipe['x'] + self.pipe_width > self.screen_width // 4:
                    nearest_pipe = pipe
                    break
                    
            if nearest_pipe is None:
                nearest_pipe = self.pipes[-1]
                
            nearest_pipe_x = nearest_pipe['x']
            nearest_pipe_y = nearest_pipe['y']
        
        # Normalize observations for better learning
        obs = np.array([
            self.bird_y / self.screen_height,  # Bird y-position (normalized)
            self.bird_velocity / self.max_velocity,  # Bird velocity (normalized)
            (nearest_pipe_x - self.screen_width // 4) / self.screen_width,  # Horizontal distance to pipe (normalized)
            nearest_pipe_y / self.screen_height  # Height of pipe gap (normalized)
        ], dtype=np.float32)
        
        return obs
    
    def render(self):
        if self.render_mode == "human":
            # Clear the screen
            self.screen.fill((135, 206, 235))  # Sky blue
            
            # Draw pipes
            for pipe in self.pipes:
                # Top pipe
                pygame.draw.rect(
                    self.screen, 
                    (0, 128, 0),  # Green
                    (pipe['x'], 0, self.pipe_width, pipe['y'])
                )
                
                # Bottom pipe
                pygame.draw.rect(
                    self.screen, 
                    (0, 128, 0),  # Green
                    (pipe['x'], pipe['y'] + self.pipe_gap, self.pipe_width, self.screen_height - pipe['y'] - self.pipe_gap)
                )
            
            # Draw bird
            pygame.draw.rect(
                self.screen, 
                (255, 255, 0),  # Yellow
                (self.screen_width // 4, self.bird_y, self.bird_width, self.bird_height)
            )
            
            # Display score
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Score: {self.score}", True, (0, 0, 0))
            self.screen.blit(score_text, (10, 10))
            
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            
    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

# For compatibility with gym.make
def make_env():
    return FlappyBirdEnv()

# Export environment for use in other files
env = FlappyBirdEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 