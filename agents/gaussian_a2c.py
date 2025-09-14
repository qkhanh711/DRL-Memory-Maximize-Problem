import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        return self.net(state)

class Gaussian_A2C:
    def __init__(self, state_dim, action_dim, max_action, device, lr=3e-4, gamma=0.99, value_coef=0.5, ent_coef=0.01, grad_norm=1.0):
        self.device = device
        self.gamma = gamma
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.grad_norm = grad_norm
        self.max_action = max_action
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
    def sample_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        logits = self.actor(state)
        dist = torch.distributions.Normal(logits, torch.ones_like(logits)*0.1)  # fixed std
        if deterministic:
            action = logits
        else:
            action = dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)
        return action.squeeze(0).detach().cpu().numpy()
    def train(self, replay_buffer, iterations, batch_size=64, log_writer=None):
        metric = {'actor_loss': [], 'critic_loss': [], 'entropy_loss': [], 'total_loss': []}
        for _ in range(iterations):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            # Compute values
            value = self.critic(state).squeeze()
            next_value = self.critic(next_state).squeeze()
            # Compute advantage and returns
            returns = reward + self.gamma * not_done * next_value
            advantage = returns - value
            # Actor loss
            logits = self.actor(state)
            dist = torch.distributions.Normal(logits, torch.ones_like(logits)*0.1)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            actor_loss = -(log_prob * advantage.detach()).mean() - self.ent_coef * entropy.mean()
            # Critic loss
            critic_loss = F.mse_loss(value, returns.detach())
            total_loss = actor_loss + self.value_coef * critic_loss
            # Optimize
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            if self.grad_norm > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            metric['actor_loss'].append(actor_loss.item())
            metric['critic_loss'].append(critic_loss.item())
            metric['entropy_loss'].append(entropy.mean().item())
            metric['total_loss'].append(total_loss.item())
        return metric
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