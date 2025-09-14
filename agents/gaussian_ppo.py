# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.model import MLP
from agents.models.common.gaussian import GaussianModel


class GaussianActor(nn.Module):
    """Gaussian Actor Network for PPO"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GaussianActor, self).__init__()
        self.mean_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.log_std_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize log_std with small values
        self.log_std_net[-1].weight.data.fill_(0.0)
        self.log_std_net[-1].bias.data.fill_(-1.0)
    
    def forward(self, cond):
        # Extract state from condition dictionary
        if isinstance(cond, dict):
            state = cond["state"]
        else:
            state = cond
            
        mean = self.mean_net(state)
        log_std = self.log_std_net(state)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
        return mean, torch.exp(log_std)


class Critic(nn.Module):
    """Value Function Network for PPO"""
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.value_model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.value_model(state)


class Gaussian_PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount=0.99,
                 tau=0.005,
                 lr=7e-3,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 clip_ratio=0.2,
                 value_clip_ratio=0.2,
                 norm_adv=True,
                 horizon_steps=1,
                 ent_coef=0.01,
                 noise_scale=0.1,
                 noise_type='gaussian',
                 epsilon=0.1,
                 ):

        # Initialize actor (Gaussian policy)
        actor_network = GaussianActor(state_dim, action_dim).to(device)
        self.actor = GaussianModel(
            network=actor_network,
            horizon_steps=horizon_steps,
            device=device
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Value function
        self.critic = Critic(state_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm
        self.clip_ratio = clip_ratio
        self.value_clip_ratio = value_clip_ratio
        self.norm_adv = norm_adv
        self.ent_coef = ent_coef

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.device = device
        
        # Noise parameters for exploration
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.epsilon = epsilon

    def compute_gae(self, rewards, values, next_value, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = next_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + gamma * lam * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return returns, advantages

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'entropy_loss': [], 'total_loss': []}
        
        for _ in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            
            # Convert to proper format for Gaussian model
            state_dict = {"state": state}
            next_state_dict = {"state": next_state}
            
            # Compute current values and next values
            with torch.no_grad():
                current_values = self.critic(state).squeeze()
                next_values = self.critic(next_state).squeeze()
                
                # Compute returns and advantages using GAE
                dones = 1 - not_done.squeeze()
                returns, advantages = self.compute_gae(
                    reward.squeeze(), 
                    current_values, 
                    next_values[-1], 
                    dones,
                    gamma=self.discount
                )
                
                # Normalize advantages
                if self.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get current policy distribution
            dist = self.actor.forward_train(state_dict, deterministic=False)
            current_log_probs = dist.log_prob(action).mean(dim=-1)
            
            # PPO loss computation
            # Policy loss with clipping
            ratio = torch.exp(current_log_probs - current_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with optional clipping
            current_values = self.critic(state).squeeze()
            if self.value_clip_ratio is not None:
                value_loss_unclipped = (current_values - returns) ** 2
                value_clipped = current_values + torch.clamp(
                    current_values - current_values.detach(),
                    -self.value_clip_ratio,
                    self.value_clip_ratio,
                )
                value_loss_clipped = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = 0.5 * ((current_values - returns) ** 2).mean()
            
            # Entropy loss for exploration
            entropy_loss = -dist.entropy().mean()
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss + self.ent_coef * entropy_loss
            
            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # Gradually reduce noise for better exploitation
            self.step_noise_decay()

            metric['ppo_loss'].append(policy_loss.item())
            metric['value_loss'].append(value_loss.item())
            metric['entropy_loss'].append(entropy_loss.item())
            metric['total_loss'].append(total_loss.item())

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state, deterministic=False, add_noise=True):
        """Sample action from the Gaussian policy with exploration noise"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_dict = {"state": state}
        
        with torch.no_grad():
            if deterministic:
                action = self.actor.forward(state_dict, deterministic=True)
            else:
                action = self.actor.forward(state_dict, deterministic=False)
                
                # Add exploration noise based on noise type
                if add_noise and not deterministic:
                    if self.noise_type == 'gaussian':
                        # Add Gaussian noise to the action
                        noise = torch.randn_like(action) * self.noise_scale
                        action = action + noise
                    elif self.noise_type == 'epsilon_greedy':
                        # Epsilon-greedy exploration
                        if np.random.random() < self.epsilon:
                            # Random action within bounds
                            action = torch.rand_like(action) * 2 * self.max_action - self.max_action
                    elif self.noise_type == 'temperature':
                        # Temperature scaling for the distribution
                        temperature = 1.0 + self.noise_scale
                        # Scale the action by temperature
                        action = action * temperature
                    elif self.noise_type == 'std_injection':
                        # Inject noise based on action magnitude
                        action_magnitude = torch.norm(action, dim=-1, keepdim=True)
                        noise_scale = self.noise_scale * action_magnitude
                        noise = torch.randn_like(action) * noise_scale
                        action = action + noise
                    else:
                        # Default: add small noise
                        noise = torch.randn_like(action) * self.noise_scale * 0.1
                        action = action + noise
        
        # Clip action to bounds
        action = torch.clamp(action, -self.max_action, self.max_action)
        
        # Add additional action noise for more exploration
        if add_noise and not deterministic and self.noise_scale > 0:
            action_np = action.cpu().data.numpy().flatten()
            action_noise = np.random.normal(0, self.noise_scale * 0.05, action_np.shape)
            action_np = action_np + action_noise
            action_np = np.clip(action_np, -self.max_action, self.max_action)
            return action_np
        
        return action.cpu().data.numpy().flatten()

    def set_noise_scale(self, noise_scale):
        """Set the noise scale for exploration"""
        self.noise_scale = noise_scale
        
    def set_epsilon(self, epsilon):
        """Set the epsilon for epsilon-greedy exploration"""
        self.epsilon = epsilon
        
    def get_noise_info(self):
        """Get current noise parameters for monitoring"""
        return {
            'noise_scale': self.noise_scale,
            'noise_type': self.noise_type,
            'epsilon': self.epsilon
        }
        
    def step_noise_decay(self):
        """Gradually reduce noise for better exploitation over time"""
        if self.noise_scale > 0.01:
            self.noise_scale *= 0.9999  # Very slow decay
        if self.epsilon > 0.01:
            self.epsilon *= 0.9999  # Very slow decay

    def save_model(self, dir, id=None):
        """Save actor and critic models"""
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        """Load actor and critic models"""
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth')) 