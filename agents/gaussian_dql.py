import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        action = normal.rsample()  # reparameterization trick
        return action, mean, std

    def sample_action(self, state):
        action, _, _ = self.sample(state)
        return action

    def deterministic_action(self, state):
        mean, _ = self.forward(state)
        return mean

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Fixed: should output 1 Q-value, not action_dim
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Fixed: should output 1 Q-value, not action_dim
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, state, action):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)

class Gaussian_DQL(object):
    def __init__(self, state_dim, action_dim, max_action, device, discount=0.99, tau=0.005, lr=3e-4, grad_norm=1.0, 
                 noise_scale=0.3, noise_type='gaussian', epsilon=0.01, policy_delay=2, **kwargs):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.grad_norm = grad_norm
        
        # Noise parameters for exploration
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.epsilon = epsilon

        self.actor = GaussianPolicy(state_dim, action_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.step = 0
        
        # Add policy delay for better stability (like TD3)
        self.policy_delay = policy_delay
        self.total_it = 0

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        metric = {'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            self.total_it += 1
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            # Critic update
            with torch.no_grad():
                next_action, _, _ = self.actor.sample(next_state)
                next_action = next_action.clamp(-self.max_action, self.max_action)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)
                target_q = reward + not_done * self.discount * target_q

            current_q1, current_q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            # Delayed policy updates (like TD3)
            if self.total_it % self.policy_delay == 0:
                # Actor update
                new_action, _, _ = self.actor.sample(state)
                new_action = new_action.clamp(-self.max_action, self.max_action)
                q1_new, q2_new = self.critic(state, new_action)
                actor_loss = -torch.min(q1_new, q2_new).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                self.actor_optimizer.step()

                # Update target networks
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                metric['actor_loss'].append(actor_loss.item())
            else:
                # Still log something for consistency
                metric['actor_loss'].append(0.0)

            # Gradually reduce noise for better exploitation
            self.step_noise_decay()

            metric['critic_loss'].append(critic_loss.item())

        return metric

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

    def sample_action(self, state, deterministic=False, add_noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic_action(state)
            else:
                action, mean, std = self.actor.sample(state)
                
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
                        else:
                            # Use mean action without additional noise since policy already has stochasticity
                            action = mean
                    elif self.noise_type == 'std_scaling':
                        # Scale the standard deviation for more exploration
                        scaled_std = std * (1.0 + self.noise_scale)
                        normal = torch.distributions.Normal(mean, scaled_std)
                        action = normal.rsample()
                    elif self.noise_type == 'temperature':
                        # Temperature scaling for the distribution
                        temperature = 1.0 + self.noise_scale
                        scaled_std = std * temperature
                        normal = torch.distributions.Normal(mean, scaled_std)
                        action = normal.rsample()
                    else:
                        # Default: use the sampled action as-is since it already has stochasticity
                        pass
        
        # Clip action to bounds
        action = action.clamp(-self.max_action, self.max_action)
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