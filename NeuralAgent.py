import torch
import torch.nn as nn
import torch.optim as optim
from utils import device
import numpy as np
import random
import pygame

def get_user_input():
    action = None
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w: action = 0
            elif event.key == pygame.K_s: action = 1
            elif event.key == pygame.K_a: action = 2
            elif event.key == pygame.K_d: action = 3
            break
    return action


class QNet(nn.Module):
    def __init__(self, input_shape, action_size):
        super(QNet, self).__init__()
        h_plane = int(input_shape[0] // 2)
        s_plane = 64
        
        self.fc1 = nn.Linear(input_shape[0], h_plane)
        self.fc2 = nn.Linear(h_plane, s_plane)
        self.fc3 = nn.Linear(s_plane, action_size)
        self.relu = nn.ReLU()

        # batch normalizatoipn
        self.bn1 = nn.BatchNorm1d(h_plane)
        self.bn2 = nn.BatchNorm1d(s_plane)
        
        # layer for skip connect
        self.skip = nn.Linear(h_plane, s_plane)
        
    
    def forward(self, state):
        x1 = self.fc1(state)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        
        # skip connection
        x1_proj = self.skip(x1)
        x2 = self.relu(x2 + x1_proj)
        
        return self.relu(self.fc3(x2))
    
class QNetConv(nn.Module):
    def __init__(self, input_shape=(15, 7, 3), output_size=4):
        super(QNetConv, self).__init__()
        
        # Transposed convolutional layer (deconv) to keep the same spatial dimensions
        self.deconv1 = nn.ConvTranspose2d(in_channels=input_shape[2],out_channels=8,kernel_size=3,stride=1,padding=1)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d( in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        
        self.relu = nn.ReLU()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers to map to output
        conv_output_size = 8 * input_shape[0] * input_shape[1]  # Calculate based on input dimensions
        self.fc1 = nn.Linear(conv_output_size, output_size)
        # self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        # x input shape: (N, 15, 7, 3)
        x = x.permute(0, 3, 1, 2)  # Rearrange to (N, 3, 15, 7) for Conv2D
        x = self.relu(self.deconv1(x))  # Output: (N, 16, 15, 7)
        x = self.relu(self.conv2(x))  # Output: (N, 32, 15, 7)

        x = self.flatten(x)  # Output: (N, 32 * 15 * 7)
        # x = self.relu(self.fc1(x))  # Output: (N, 128)
        # x = self.fc2(x)  # Output: (N, 4)
        x = self.fc1(x)
        return x

class DQLAgent:
    def __init__(self, state_size, action_size, learning_rate, gamma, epsilon_decay, num_of_rewards=1, epsilon_min=0.05, epsilon_max=1, use_convolution=False):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.use_convolution = use_convolution
        self.local_rng = np.random.default_rng()
        self.max_injection_count = 16
        Model = QNetConv if use_convolution else QNet
        self.q_network = Model(state_size, action_size).to(device)
        self.target_network = Model(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        # Experience replay buffer
        self.memory = []
        self.max_mem_size = 25000
        self.batch_size = 128
        self.memory_hash = []

    def reduce_memory(self):
        if len(self.memory) > self.batch_size:
            indices = np.random.choice(range(len(self.memory)), self.batch_size)
            self.memory = np.array(self.memory, dtype=object)[indices].tolist()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.max_mem_size:  # Limit buffer size
            i = np.random.randint(len(self.memory))
            self.memory.pop(i)

    def act(self, state):
        
        with torch.no_grad():
            self.q_network.eval()

            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            probs = torch.nn.functional.softmax(q_values)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()# / len(probs[0])

            self.q_network.train()

        if self.local_rng.uniform(0, 1) < self.epsilon:
            choice = self.local_rng.choice(self.action_size)
        else:
            choice = torch.argmax(q_values).item()

        return choice, entropy
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return -1
        

        _, _, rewards, _, dones = zip(*self.memory)
        rewards = np.array(rewards)
        indices = np.where(rewards > 0)[0]
        np.random.shuffle(indices)
        indices = indices[:self.max_injection_count]

        chosen = [i for i in indices if np.random.rand() > .7]
        
        batch = random.sample(self.memory, self.batch_size - len(chosen))
        states, actions, rewards, next_states, dones = zip(*batch)

        if len(chosen) > 0:
            _batch = [self.memory[i] for i in chosen]
            _states, _actions, _rewards, _next_states, _dones = zip(*_batch)
            states = np.array(states + _states)
            actions = np.array(actions + _actions)
            rewards = np.array(rewards + _rewards)
            next_states = np.array(next_states + _next_states)
            dones = np.array(dones + _dones)
            
            new_indices = list(range(len(states)))
            np.random.shuffle(new_indices)
            states = states[new_indices]
            actions = actions[new_indices]
            rewards = rewards[new_indices]
            next_states = next_states[new_indices]
            dones = dones[new_indices]
        
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
        q_values = self.q_network(states)

        # Compute loss and update the network
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        
    def adjust_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
