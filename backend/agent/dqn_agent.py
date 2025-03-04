import numpy as np
import tensorflow as tf
from collections import deque
import random
import os

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.update_target_frequency = 10  # how often to update target network
        
        # Memory for experience replay
        self.memory = deque(maxlen=10000)
        
        # Main model (trained every step)
        self.model = self._build_model()
        
        # Target model (used for prediction)
        self.target_model = self._build_model()
        self.update_target_model()
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []
        
        # Model saving
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    def _build_model(self):
        """Neural Network for Deep-Q learning Model."""
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """Return action based on epsilon-greedy policy"""
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train on random batch from memory"""
        if len(self.memory) < batch_size:
            return 0
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        # Q(s,a) = r + γ * max(Q(s',a'))
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon_history.append(self.epsilon)
        
        return loss
    
    def load(self, name):
        """Load model weights"""
        model_path = os.path.join(self.model_dir, name)
        self.model.load_weights(model_path)
        self.update_target_model()
    
    def save(self, name):
        """Save model weights"""
        model_path = os.path.join(self.model_dir, name)
        self.model.save_weights(model_path)
    
    def get_metrics(self):
        """Return training metrics"""
        return {
            'loss_history': self.loss_history,
            'epsilon_history': self.epsilon_history,
            'reward_history': self.reward_history
        } 