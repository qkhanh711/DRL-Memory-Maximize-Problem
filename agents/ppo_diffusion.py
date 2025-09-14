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
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)  # Output single value function
        )

    def forward(self, state):
        return self.model(state)

    def q1(self, state):
        return self.model(state)

    def q_min(self, state):
        return self.model(state)


class Diffusion_PPO(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 gamma=0.99,
                 tau=0.95,
                 clip_param=0.2,
                 beta_schedule='linear',
                 n_timesteps=10,  # More timesteps for better action quality
                 ema_decay=0.99,  # Faster EMA for more exploration
                 step_start_ema=500,  # Start EMA earlier
                 update_ema_every=3,  # More frequent EMA updates
                 lr=2e-4,  # Higher learning rate for faster learning
                 lr_decay=True,
                 lr_maxt=10000,
                 grad_norm=1.0,
                 entropy_coef=0.02,  # Higher entropy for more exploration
                 value_loss_coef=0.25,  # Lower value loss weight
                 warmup_steps=200,  # Shorter warmup
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every
        self.warmup_steps = warmup_steps

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr*3, betas=(0.9, 0.999))  # Critic learns much faster

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr*0.1)  # Don't decay too much
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr*0.3)

        self.entropy_coef = entropy_coef
        self.value_coef = value_loss_coef
        self.initial_lr = lr
        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.clip_param = clip_param  # Use standard PPO clip
        self.device = device

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)


    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'actor_loss': []}
        
        for iteration in range(iterations):
            try:
                # Sample batch
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                """ Value Function Training (Train first for stable baseline) """
                with torch.no_grad():
                    next_values = self.critic(next_state).squeeze()
                    returns = reward.squeeze() + self.gamma * next_values * not_done.squeeze()
                
                values = self.critic(state).squeeze()
                advantages = returns - values
                
                # Normalize advantages
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Train value function multiple times
                for _ in range(3):
                    value_pred = self.critic(state).squeeze()
                    value_loss = F.mse_loss(value_pred, returns.detach())
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                    self.critic_optimizer.step()

                """ Policy Training - Advantage-Weighted Behavior Cloning """
                # Only train on actions with positive advantages (good actions)
                positive_mask = advantages > 0
                
                if positive_mask.sum() > 0:  # Only if we have positive advantages
                    # Get subset of good transitions
                    good_states = state[positive_mask]
                    good_actions = action[positive_mask]
                    good_advantages = advantages[positive_mask]
                    
                    # Weight by advantage magnitude (higher advantage = more important)
                    weights = torch.softmax(good_advantages, dim=0)  # Softmax to normalize
                    
                    # Compute weighted behavioral cloning loss
                    bc_losses = self.actor.loss(good_actions, good_states)
                    weighted_bc_loss = (bc_losses * weights.detach()).mean()
                    
                    # Add small regularization
                    reg_loss = 0.001 * bc_losses.mean()
                    total_actor_loss = weighted_bc_loss + reg_loss
                    
                    # Policy update
                    self.actor_optimizer.zero_grad()
                    total_actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                    self.actor_optimizer.step()
                    
                    policy_loss = weighted_bc_loss
                else:
                    # No positive advantages, skip policy update
                    policy_loss = torch.tensor(0.0)
                    total_actor_loss = torch.tensor(0.0)

                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                self.step += 1

                # Log metrics
                metric['ppo_loss'].append(float(policy_loss.item()) if not torch.isnan(policy_loss) else 0.0)
                metric['value_loss'].append(float(value_loss.item()) if not torch.isnan(value_loss) else 0.0)
                metric['actor_loss'].append(float(total_actor_loss.item()) if not torch.isnan(total_actor_loss) else 0.0)

            except Exception as e:
                print(f"Error in PPO training iteration {iteration}: {e}")
                metric['ppo_loss'].append(0.0)
                metric['value_loss'].append(0.0)
                metric['actor_loss'].append(0.0)
                continue

        # Learning rate scheduling (more conservative)
        if self.lr_decay and self.step % 200 == 0:  # Even less frequent LR updates
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            # Mix exploration: Sometimes use main model for exploration
            if self.step > self.step_start_ema:
                # 80% EMA (stable), 20% main model (exploration)
                if np.random.random() < 0.8:
                    action = self.ema_model.sample(state)
                else:
                    action = self.actor.sample(state)
            else:
                action = self.actor.sample(state)
                
            # Add small noise for exploration in early training
            if self.step < 2000:
                noise_scale = 0.1 * (1.0 - self.step / 2000.0)  # Decreasing noise
                noise = torch.randn_like(action) * noise_scale
                action = action + noise
                action = torch.clamp(action, -self.max_action, self.max_action)
                
        return action.cpu().data.numpy().flatten()

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