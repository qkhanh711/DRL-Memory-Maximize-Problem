import numpy as np
import torch
from collections import deque
import random


class ReplayBuffer:
    """Replay buffer for storing and sampling transitions"""
    
    def __init__(self, capacity=100000, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, next_state, reward, done):
        """Add a transition to the buffer"""
        self.buffer.append((state, action, next_state, reward, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        
        # Convert to tensors
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        
        return state, action, next_state, reward, done
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer"""
    
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, device='cpu'):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
    def push(self, state, action, next_state, reward, done):
        """Add a transition to the buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, next_state, reward, done)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of transitions with priorities"""
        if len(self.buffer) == 0:
            return None, None, None, None, None, None, None
            
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        state, action, next_state, reward, done = zip(*batch)
        
        # Convert to tensors
        state = torch.FloatTensor(np.array(state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.array(done)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        return state, action, next_state, reward, done, weights, indices
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)