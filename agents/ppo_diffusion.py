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
                 beta_diffusion=0.5,  # Weight for diffusion loss
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)

        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

        # Initialize old policy for PPO ratio computation
        self.actor_old = copy.deepcopy(self.actor)
        
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
        self.beta_diffusion = beta_diffusion
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


    def compute_gae(self, rewards, values, next_values, not_dones):
        """
        Compute Generalized Advantage Estimation (GAE)
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        # Compute GAE backwards
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * not_dones[t] - values[t]
            # Clamp delta to prevent explosion
            delta = torch.clamp(delta, -10.0, 10.0)
            advantages[t] = last_advantage = delta + self.gamma * self.tau * not_dones[t] * last_advantage
        
        returns = advantages + values
        return advantages, returns

    def compute_ppo_loss(self, state, action, advantages):
        """
        Compute PPO clipped loss with diffusion policy
        """
        # Get log prob from current policy (using diffusion loss as proxy)
        current_loss = self.actor.loss(action, state)
        
        # Get log prob from old policy
        with torch.no_grad():
            old_loss = self.actor_old.loss(action, state)
        
        # Compute ratio: pi(a|s) / pi_old(a|s)
        # Since we use loss (negative log prob), ratio = exp(old_loss - current_loss)
        # Clamp the exponent to prevent overflow/underflow
        log_ratio = torch.clamp(old_loss - current_loss, -10.0, 10.0)
        ratio = torch.exp(log_ratio)
        
        # Check for NaN and replace with 1.0 (no change)
        ratio = torch.where(torch.isnan(ratio), torch.ones_like(ratio), ratio)
        ratio = torch.clamp(ratio, 0.1, 10.0)  # Prevent extreme ratios
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        
        return ppo_loss, ratio

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'ppo_loss': [], 'value_loss': [], 'actor_loss': [], 'diffusion_loss': []}
        
        for iteration in range(iterations):
            try:
                # ============================================
                # Trajectory Collection: Sample batch
                # ============================================
                state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

                # ============================================
                # Value Function Update (train first for stable baseline)
                # ============================================
                with torch.no_grad():
                    next_values = self.critic(next_state).squeeze()
                    returns = reward.squeeze() + self.gamma * next_values * not_done.squeeze()
                
                # Train value function multiple times
                value_loss = None
                for _ in range(3):
                    value_pred = self.critic(state).squeeze()
                    value_loss = F.mse_loss(value_pred, returns.detach())
                    
                    if torch.isnan(value_loss) or torch.isinf(value_loss):
                        print(f"Warning: Invalid value loss at iteration {iteration}, skipping")
                        break
                    
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_norm)
                    self.critic_optimizer.step()

                # Skip if value training failed
                if value_loss is None or torch.isnan(value_loss) or torch.isinf(value_loss):
                    metric['ppo_loss'].append(0.0)
                    metric['diffusion_loss'].append(0.0)
                    metric['value_loss'].append(0.0)
                    metric['actor_loss'].append(0.0)
                    continue

                # ============================================
                # Advantage Estimation: Compute GAE
                # ============================================
                with torch.no_grad():
                    values = self.critic(state).squeeze()
                    next_values = self.critic(next_state).squeeze()
                    
                    # Compute advantages using GAE
                    advantages, returns = self.compute_gae(
                        reward.squeeze(), 
                        values, 
                        next_values, 
                        not_done.squeeze()
                    )
                    
                    # Normalize advantages for stability
                    if advantages.std() > 1e-8:
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                    else:
                        advantages = advantages - advantages.mean()

                # ============================================
                # Policy Update (Diffusion Actor with PPO)
                # ============================================
                
                # Compute diffusion behavior cloning loss
                diffusion_loss = self.actor.loss(action, state)
                
                if torch.isnan(diffusion_loss).any() or torch.isinf(diffusion_loss).any():
                    print(f"Warning: Invalid diffusion loss at iteration {iteration}, skipping policy update")
                    metric['ppo_loss'].append(0.0)
                    metric['diffusion_loss'].append(0.0)
                    metric['value_loss'].append(float(value_loss.item()))
                    metric['actor_loss'].append(0.0)
                    continue
                
                # Compute PPO clipped loss
                ppo_loss, ratio = self.compute_ppo_loss(state, action, advantages.detach())
                
                if torch.isnan(ppo_loss) or torch.isinf(ppo_loss):
                    print(f"Warning: Invalid PPO loss at iteration {iteration}, using diffusion loss only")
                    # Fall back to diffusion loss only
                    total_actor_loss = diffusion_loss.mean()
                    ppo_loss = torch.tensor(0.0)
                else:
                    # Total actor loss: L_actor = L_PPO + beta * L_diffusion
                    total_actor_loss = ppo_loss + self.beta_diffusion * diffusion_loss.mean()
                
                # Policy update via gradient descent
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm)
                self.actor_optimizer.step()

                # ============================================
                # Update old policy: theta_old <- theta
                # ============================================
                if iteration % 10 == 0:  # Update old policy periodically
                    self.actor_old.load_state_dict(self.actor.state_dict())

                # EMA update
                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                self.step += 1

                # Log metrics (safely handle NaN)
                metric['ppo_loss'].append(float(ppo_loss.item()) if not (torch.isnan(ppo_loss) or torch.isinf(ppo_loss)) else 0.0)
                metric['diffusion_loss'].append(float(diffusion_loss.mean().item()) if not (torch.isnan(diffusion_loss).any() or torch.isinf(diffusion_loss).any()) else 0.0)
                metric['value_loss'].append(float(value_loss.item()) if not (torch.isnan(value_loss) or torch.isinf(value_loss)) else 0.0)
                metric['actor_loss'].append(float(total_actor_loss.item()) if not (torch.isnan(total_actor_loss) or torch.isinf(total_actor_loss)) else 0.0)

            except Exception as e:
                print(f"Error in PPO training iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                metric['ppo_loss'].append(0.0)
                metric['diffusion_loss'].append(0.0)
                metric['value_loss'].append(0.0)
                metric['actor_loss'].append(0.0)
                continue

        # Learning rate scheduling
        if self.lr_decay and self.step % 200 == 0:
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