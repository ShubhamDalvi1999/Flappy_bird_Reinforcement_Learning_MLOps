import numpy as np
import tensorflow as tf
from collections import deque
import random

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
        
    def _build_model(self):
        """Neural Network for Deep Q-learning"""
        with tf.device('/gpu:0'):  # Use GPU if available
            model = tf.keras.Sequential()
            model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
            model.add(tf.keras.layers.Dense(24, activation='relu'))
            model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model
    
    def update_target_model(self):
        """Copy weights from main model to target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, explore=True):
        """Return action based on epsilon-greedy policy"""
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """Train model using experience replay"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size))
        targets = np.zeros((batch_size, self.action_size))
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            
            if done:
                target = reward
            else:
                # Double DQN: select action using main model, evaluate with target model
                next_action = np.argmax(self.model.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0])
                target = reward + self.gamma * self.target_model.predict(np.reshape(next_state, [1, self.state_size]), verbose=0)[0][next_action]
            
            target_f = self.model.predict(np.reshape(state, [1, self.state_size]), verbose=0)
            target_f[0][action] = target
            targets[i] = target_f[0]
        
        # Train the model
        history = self.model.fit(states, targets, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def load(self, name):
        """Load model weights"""
        self.model.load_weights(name)
        self.update_target_model()
        
    def save(self, name):
        """Save model weights"""
        self.model.save_weights(name)
        
    def get_epsilon(self):
        """Return current epsilon value"""
        return self.epsilon 