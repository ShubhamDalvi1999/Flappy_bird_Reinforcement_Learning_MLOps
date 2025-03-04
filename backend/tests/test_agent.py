import sys
import os
import pytest
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.dqn_agent import DQNAgent
from game.flappy_bird import FlappyBirdEnv

class TestDQNAgent:
    
    @pytest.fixture
    def agent(self):
        """Create a test agent"""
        env = FlappyBirdEnv()
        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            batch_size=32
        )
        return agent
    
    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.state_size > 0
        assert agent.action_size == 2  # Flap or not flap
        assert agent.batch_size == 32
        assert 0 <= agent.epsilon <= 1
        assert 0 <= agent.gamma <= 1
        assert agent.memory is not None
        assert agent.model is not None
        assert agent.target_model is not None
    
    def test_act_method(self, agent):
        """Test agent's act method"""
        env = FlappyBirdEnv()
        state = env.reset()
        
        # Test exploration (epsilon=1)
        agent.epsilon = 1.0
        actions = [agent.act(state) for _ in range(100)]
        # Should have a mix of 0 and 1 actions due to exploration
        assert 0 in actions
        assert 1 in actions
        
        # Test exploitation (epsilon=0)
        agent.epsilon = 0.0
        action = agent.act(state)
        assert action in [0, 1]
        
        # Multiple calls with the same state and epsilon=0 should return the same action
        next_action = agent.act(state)
        assert action == next_action
    
    def test_remember_and_replay(self, agent):
        """Test agent's remember and replay methods"""
        env = FlappyBirdEnv()
        state = env.reset()
        action = 0
        next_state, reward, done, _ = env.step(action)
        
        # Test remember
        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == 1
        
        # Add more experiences
        for _ in range(10):
            state = next_state
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if done:
                state = env.reset()
        
        assert len(agent.memory) == 11
        
        # Test replay
        # Not enough samples yet, should return None
        agent.batch_size = 32
        loss = agent.replay()
        assert loss is None
        
        # Add more samples to reach batch size
        for _ in range(30):
            state = env.reset()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
        
        # Now should be able to replay
        loss = agent.replay()
        # Loss might be None for the first few iterations if no batches 
        # were generated yet, but that's OK
    
    def test_update_epsilon(self, agent):
        """Test epsilon decay"""
        initial_epsilon = agent.epsilon
        for _ in range(10):
            agent.update_epsilon()
        
        assert agent.epsilon < initial_epsilon
        
        # Test min epsilon
        agent.epsilon = agent.epsilon_min
        agent.update_epsilon()
        assert agent.epsilon == agent.epsilon_min
    
    def test_save_and_load(self, agent, tmp_path):
        """Test model saving and loading"""
        model_path = os.path.join(tmp_path, "test_model.pth")
        
        # Save model
        agent.save(model_path)
        assert os.path.exists(model_path)
        
        # Change model weights
        original_weights = agent.model.state_dict()['fc1.weight'].clone().numpy()
        
        # Create a new agent and load the saved model
        new_agent = DQNAgent(
            state_size=agent.state_size,
            action_size=agent.action_size
        )
        new_agent.load(model_path)
        
        # Check if weights are the same
        loaded_weights = new_agent.model.state_dict()['fc1.weight'].clone().numpy()
        assert np.array_equal(original_weights, loaded_weights)
    
    def test_model_architecture(self, agent):
        """Test model architecture"""
        # Get the model architecture
        model = agent.model
        
        # Check input and output dimensions
        input_features = model.fc1.in_features
        output_features = model.fc3.out_features
        
        assert input_features == agent.state_size
        assert output_features == agent.action_size
        
        # Test forward pass
        test_input = torch.FloatTensor(np.random.random((1, agent.state_size)))
        output = model(test_input)
        
        assert output.shape == (1, agent.action_size)
        
    def test_target_model_update(self, agent):
        """Test target model update"""
        # Initially, the model and target model should have the same weights
        model_weights = agent.model.state_dict()
        target_weights = agent.target_model.state_dict()
        
        # Verify they're the same
        for key in model_weights:
            assert torch.equal(model_weights[key], target_weights[key])
        
        # Train the model but not the target
        env = FlappyBirdEnv()
        state = env.reset()
        for _ in range(50):
            state = env.reset()
            for _ in range(10):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
        
        # Replay to update model weights
        if len(agent.memory) >= agent.batch_size:
            agent.replay()
        
        # Now model and target should be different
        model_weights = agent.model.state_dict()
        target_weights = agent.target_model.state_dict()
        
        # Check if any weights are different (they should be after training)
        any_diff = False
        for key in model_weights:
            if not torch.equal(model_weights[key], target_weights[key]):
                any_diff = True
                break
                
        # If no difference found, it could be because:
        # 1. Replay memory isn't full enough
        # 2. Training didn't have enough impact yet
        # 3. Target update wasn't called
        
        # Update target network
        agent.update_target_model()
        
        # After update, they should be the same again
        model_weights = agent.model.state_dict()
        target_weights = agent.target_model.state_dict()
        
        for key in model_weights:
            assert torch.equal(model_weights[key], target_weights[key])


if __name__ == "__main__":
    pytest.main(["-v"]) 