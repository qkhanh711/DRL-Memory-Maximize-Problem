# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1)  # Output single value function
        )

    def forward(self, state):
        return self.model(state)

    def q1(self, state):
        return self.model(state)

    def q_min(self, state):
        return self.model(state)


class Diffusion_A2C(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 gamma=0.99,
                 lr=1e-4,              # Much smaller learning rate
                 beta_schedule='linear',
                 n_timesteps=20,       # Fewer timesteps for stability
                 ema_decay=0.995,
                 step_start_ema=2000,  # Later EMA start
                 update_ema_every=10,  # Less frequent EMA updates
                 grad_norm=0.25,       # Smaller gradient clipping
                 entropy_coef=0.01,    # Add entropy regularization
                 value_loss_coef=0.25,
                 lr_decay=False,
                 max_grad_norm=0.5,    # Smaller gradient norm
                 ): # Smaller value loss weight

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-8)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr*2, eps=1e-8)

        # Add learning rate schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    def compute_returns(self, rewards, values, dones, next_value):
        """Compute returns using simple bootstrapping"""
        returns = []
        R = next_value
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)
        return returns

    def train(self, replay_buffer, iterations, batch_size=64):
        metric = {'actor_loss': [], 'critic_loss': [], 'total_loss': []}
        
        for _ in range(iterations):
            try:
                # Sample batch
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
                
                # Compute returns and advantages with more stability
                with torch.no_grad():
                    values = self.critic(state).squeeze()
                    next_values = self.critic(next_state).squeeze()
                    
                    # TD targets
                    returns = reward + self.gamma * next_values * not_done
                    
                    # Advantages with normalization
                    advantages = returns - values
                    # More aggressive normalization for A2C
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
                    advantages = torch.clamp(advantages, -10.0, 10.0)
                
                """ Actor Training """
                # Compute policy loss
                action_loss = self.actor.loss(action, state)
                action_loss = torch.clamp(action_loss, max=10.0)  # Prevent explosion
                
                # Policy gradient with entropy
                policy_loss = (action_loss * advantages.detach()).mean()
                
                # Add entropy regularization
                if hasattr(self.actor, 'get_entropy'):
                    entropy = self.actor.get_entropy(state)
                    policy_loss -= self.entropy_coef * entropy.mean()
                
                # Check for valid loss
                if torch.isnan(policy_loss) or torch.isinf(policy_loss) or abs(policy_loss.item()) > 100:
                    # print(f"Warning: Invalid policy_loss {policy_loss}, skipping")
                    metric['actor_loss'].append(0.0)
                    metric['critic_loss'].append(0.0)
                    metric['total_loss'].append(0.0)
                    continue
                
                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
                self.actor_optimizer.step()
                
                """ Critic Training """
                values_pred = self.critic(state).squeeze()
                if values_pred.dim() != returns.dim():
                    returns = returns.squeeze()
                
                critic_loss = F.mse_loss(values_pred, returns.detach()) * self.value_loss_coef
                
                if torch.isnan(critic_loss) or torch.isinf(critic_loss) or abs(critic_loss.item()) > 100:
                    # print(f"Warning: Invalid critic_loss {critic_loss}, skipping")
                    metric['actor_loss'].append(float(policy_loss.item()))
                    metric['critic_loss'].append(0.0)
                    metric['total_loss'].append(float(policy_loss.item()))
                    continue
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm)
                self.critic_optimizer.step()
                
                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                
                self.step += 1
                
                # Log metrics
                metric['actor_loss'].append(float(policy_loss.item()))
                metric['critic_loss'].append(float(critic_loss.item()))
                metric['total_loss'].append(float(policy_loss.item() + critic_loss.item()))
                
            except Exception as e:
                print(f"Error in A2C training: {e}")
                metric['actor_loss'].append(0.0)
                metric['critic_loss'].append(0.0)
                metric['total_loss'].append(0.0)
                continue
    
        # Update learning rates
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=50, dim=0)
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            # Use value function to guide action selection
            values = self.critic(state_rpt).flatten()
            idx = torch.multinomial(F.softmax(values, dim=0), 1)
        return action[idx].cpu().data.numpy().flatten()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))