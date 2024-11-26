import torch
import torch.nn as nn
import torch.optim as optim
from utils import device
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size[0], state_size[0] // 2)
        self.fc2 = nn.Linear(state_size[0] // 2, 128)
        self.fc3 = nn.Linear(128, action_size)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class QNetworkConv(nn.Module):
    def __init__(self, input_shape=(15, 7, 3), output_size=4):
        super(QNetworkConv, self).__init__()
        
        # Transposed convolutional layer (deconv) to keep the same spatial dimensions
        self.deconv1 = nn.ConvTranspose2d(in_channels=input_shape[2],out_channels=16,kernel_size=3,stride=1,padding=1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d( in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers to map to output
        conv_output_size = 8 * input_shape[0] * input_shape[1]  # Calculate based on input dimensions
        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        # x input shape: (N, 15, 7, 3)
        x = x.permute(0, 3, 1, 2)  # Rearrange to (N, 3, 15, 7) for Conv2D
        x = self.relu(self.deconv1(x))  # Output: (N, 16, 15, 7)
        x = self.relu(self.conv2(x))  # Output: (N, 32, 15, 7)

        x = self.flatten(x)  # Output: (N, 32 * 15 * 7)
        x = self.relu(self.fc1(x))  # Output: (N, 128)
        x = self.fc2(x)  # Output: (N, 4)
        return x

class DQLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.01, use_convolution=False):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.use_convolution = use_convolution
        # Q-Networks
        Model = QNetworkConv if use_convolution else QNetwork
        self.q_network = Model(state_size, action_size).to(device)
        self.target_network = Model(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = []
        self.max_mem_size = min(10000, 3 * np.prod(self.state_size) * self.action_size)
        self.batch_size = 32
        self.update_target_frequency = 100  # Update target network every X steps
        self.memory_hash = []

    def store_transition(self, state, action, reward, next_state, done):
        h = hash((tuple(state.flatten()), action))
        if h in self.memory_hash:
            # print(f"Hash {h} exists")
            return
        self.memory_hash.append(h)
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_mem_size:  # Limit buffer size
            self.memory.pop(0)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()

    def train(self):
        if len(self.memory) < self.batch_size:
            return -1
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            target_q_values = rewards + (1 - dones) * self.gamma * torch.max(next_q_values, dim=1)[0]
        
        # Compute current Q-values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        # Compute loss and update the network
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
