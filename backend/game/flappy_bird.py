import os
import pygame
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import io
import base64

pygame.init()

class Bird:
    IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", f"bird{i}.png"))) for i in range(1, 4)]
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1
        d = self.vel * self.tick_count + 1.5 * self.tick_count**2

        if d >= 16:
            d = 16
        if d < 0:
            d -= 2

        self.y = self.y + d

        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.img_count += 1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt <= -80:
            self.img = self.IMGS[1]
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Pipe:
    GAP = 200
    VEL = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        self.top = 0
        self.bottom = 0
        self.PIPE_TOP = pygame.transform.flip(pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png"))), False, True)
        self.PIPE_BOTTOM = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))

        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.VEL

    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        return bool(t_point or b_point)

class Base:
    VEL = 5
    WIDTH = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png"))).get_width()
    IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

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
        
        # Initialize Pygame
        self.win_width = 600
        self.win_height = 800
        self.win = pygame.display.set_mode((self.win_width, self.win_height))
        pygame.display.set_caption("Flappy Bird")
        
        # Load game assets
        self.bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")), (self.win_width, self.win_height))
        
        # Initialize game objects
        self.bird = None
        self.base = None
        self.clock = pygame.time.Clock()
        
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
        
        self.bird = Bird(230, 350)
        self.pipes = [Pipe(700)]
        self.base = Base(730)
        
        state = self._get_state()
        return state, {}
    
    def step(self, action):
        self.steps += 1
        
        # Apply action (flap)
        if action == 1:
            self.bird.jump()
        
        # Move bird and update game objects
        self.bird.move()
        self.base.move()

        # Add new pipe
        if len(self.pipes) > 0 and 600 - self.pipes[-1].x > 300:
            self.pipes.append(Pipe(600))

        # Move pipes and remove off-screen pipes
        rem = []
        for pipe in self.pipes:
            pipe.move()
            
            # Check for collision
            if pipe.collide(self.bird):
                done = True
                reward = -1
                break
            
            # Remove off-screen pipes
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)
                
            # Check if bird passed pipe
            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                self.score += 1
                reward = 1

        # Remove passed pipes
        for r in rem:
            self.pipes.remove(r)

        # Check for ground collision
        if self.bird.y + self.bird.img.get_height() >= 730 or self.bird.y < 0:
            done = True
            reward = -1

        state = self._get_state()
        info = {'score': self.score}
        
        return state, reward, done, False, info
    
    def _get_state(self):
        if len(self.pipes) > 0:
            next_pipe = self.pipes[0]
            return np.array([
                self.bird.y,
                self.bird.vel,
                next_pipe.height,
                next_pipe.x - self.bird.x
            ], dtype=np.float32)
        return np.zeros(4, dtype=np.float32)
    
    def render(self):
        # Draw background
        self.win.blit(self.bg_img, (0, 0))
        
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.win)
            
        # Draw base and bird
        self.base.draw(self.win)
        self.bird.draw(self.win)
        
        # Draw score
        score_font = pygame.font.SysFont("comicsans", 50)
        score_label = score_font.render(f"Score: {self.score}", 1, (255, 255, 255))
        self.win.blit(score_label, (self.win_width - score_label.get_width() - 15, 10))
        
        pygame.display.update()
        self.clock.tick(30)  # 30 FPS
        
        # Convert surface to PNG for transmission
        # Create a byte buffer and save the pygame surface as a PNG
        buffer = io.BytesIO()
        pygame.image.save(self.win, buffer, "PNG")
        buffer.seek(0)
        
        # Return the raw bytes for the caller to encode if needed
        return buffer.read()
    
    def close(self):
        pygame.quit()

# For compatibility with gym.make
def make_env():
    return FlappyBirdEnv()

# Export environment for use in other files
env = FlappyBirdEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n 