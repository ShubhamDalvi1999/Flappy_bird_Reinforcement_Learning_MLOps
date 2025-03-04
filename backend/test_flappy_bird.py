import pytest
import numpy as np
import tensorflow as tf
from environment import FlappyBirdEnv, env, state_size, action_size
from agent import DQNAgent

# Tests for the Flappy Bird environment
class TestFlappyBirdEnvironment:
    def test_environment_initialization(self):
        """Test if the environment can be initialized correctly"""
        test_env = FlappyBirdEnv()
        assert test_env is not None
        assert test_env.action_space.n == 2  # 0: do nothing, 1: flap
        assert test_env.observation_space.shape[0] == 4  # state size should be 4

    def test_environment_reset(self):
        """Test if the environment can be reset correctly"""
        test_env = FlappyBirdEnv()
        state, info = test_env.reset()
        
        # Check state properties
        assert state is not None
        assert state.shape == (4,)  # should be a 4-dimensional state
        assert isinstance(state, np.ndarray)
        
        # Check if bird position is initialized at the center
        assert np.isclose(state[0], 0.5, atol=0.1)  # bird y position (normalized)
        assert np.isclose(state[1], 0.0, atol=0.1)  # bird velocity (normalized)

    def test_environment_step(self):
        """Test if the environment step function works correctly"""
        test_env = FlappyBirdEnv()
        state, _ = test_env.reset()
        
        # Test do nothing action (0)
        next_state, reward, terminated, truncated, info = test_env.step(0)
        assert next_state is not None
        assert next_state.shape == (4,)
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Bird should move downward with no action (due to gravity)
        assert next_state[1] > state[1]  # Velocity should increase (gravity)
        
        # Test flap action (1)
        state, _ = test_env.reset()
        before_velocity = state[1]
        next_state, _, _, _, _ = test_env.step(1)
        after_velocity = next_state[1]
        
        # Flap should make the bird go up (negative velocity)
        assert after_velocity < before_velocity

    def test_collision_detection(self):
        """Test collision detection in the environment"""
        test_env = FlappyBirdEnv()
        test_env.reset()
        
        # Force bird to go out of bounds (to ceiling)
        test_env.bird_y = -10
        assert test_env._check_collision() is True
        
        # Force bird to go out of bounds (to ground)
        test_env.bird_y = test_env.screen_height + 10
        assert test_env._check_collision() is True
        
        # Reset to normal position
        test_env.reset()
        assert test_env._check_collision() is False

# Tests for the DQN Agent
class TestDQNAgent:
    def test_agent_initialization(self):
        """Test if the agent can be initialized correctly"""
        agent = DQNAgent(state_size, action_size)
        assert agent is not None
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert 0 < agent.epsilon <= 1.0  # Epsilon should be in valid range

    def test_model_architecture(self):
        """Test if the model has the correct architecture"""
        agent = DQNAgent(state_size, action_size)
        
        # Check model summary to ensure correct architecture
        assert len(agent.model.layers) >= 3  # Input, hidden, output
        
        # Check input shape
        assert agent.model.layers[0].input_shape == (None, state_size)
        
        # Check output shape
        assert agent.model.layers[-1].output_shape == (None, action_size)

    def test_act_function(self):
        """Test if the agent can select actions"""
        agent = DQNAgent(state_size, action_size)
        state = np.random.random(state_size)
        
        # Test deterministic action (explore=False)
        action = agent.act(state, explore=False)
        assert 0 <= action < action_size
        
        # Test with exploration (explore=True)
        # Run multiple times to ensure we're testing both random and predicted actions
        actions = [agent.act(state, explore=True) for _ in range(100)]
        assert all(0 <= a < action_size for a in actions)

    def test_memory_functionality(self):
        """Test if experience replay memory works correctly"""
        agent = DQNAgent(state_size, action_size)
        
        # Check initial memory state
        assert len(agent.memory) == 0
        
        # Add experiences
        for _ in range(5):
            state = np.random.random(state_size)
            action = np.random.randint(0, action_size)
            reward = np.random.random()
            next_state = np.random.random(state_size)
            done = np.random.choice([True, False])
            
            agent.remember(state, action, reward, next_state, done)
        
        # Check memory size
        assert len(agent.memory) == 5
        
        # Check memory content type
        state, action, reward, next_state, done = agent.memory[0]
        assert state.shape == (state_size,)
        assert isinstance(action, (int, np.integer))
        assert isinstance(reward, (float, np.floating))
        assert next_state.shape == (state_size,)
        assert isinstance(done, bool)

    def test_target_model_update(self):
        """Test if the target model can be updated correctly"""
        agent = DQNAgent(state_size, action_size)
        
        # Get initial weights
        initial_weights = agent.target_model.get_weights()[0].copy()
        
        # Modify main model
        agent.model.layers[0].kernel.assign(agent.model.layers[0].kernel * 0.5)
        
        # Target model should still have original weights
        assert not np.array_equal(agent.model.get_weights()[0], initial_weights)
        
        # Update target model
        agent.update_target_model()
        
        # Now target model should have the same weights as the main model
        assert np.array_equal(agent.target_model.get_weights()[0], agent.model.get_weights()[0])

if __name__ == '__main__':
    pytest.main(["-v"]) 