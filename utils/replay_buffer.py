import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=1000000, device='cpu'):
        self.max_size = max_size
        self.device = device
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, next_state, reward, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states, actions, next_states, rewards, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        not_dones = torch.FloatTensor(1 - np.array(dones)).unsqueeze(1).to(self.device)
        
        return states, actions, next_states, rewards, not_dones
    
    def size(self):
        return len(self.buffer)