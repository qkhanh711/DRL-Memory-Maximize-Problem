#!/usr/bin/env python3
"""
Training script for DRL Memory Maximization Problem
Supports multiple agent types: A2C, BC, Gaussian A2C, Gaussian DQL, Gaussian PPO, PPO Diffusion, QL Diffusion
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import wandb
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import environment and agents
from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
from agents.a2c_diffusion import Diffusion_A2C
from agents.bc_diffusion import Diffusion_BC
from agents.gaussian_a2c import Gaussian_A2C
from agents.gaussian_dql import Gaussian_DQL
from agents.gaussian_ppo import Gaussian_PPO
from agents.ppo_diffusion import Diffusion_PPO
from agents.ql_diffusion import Diffusion_QL
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class Trainer:
    """Main trainer class for DRL Memory Maximization Problem"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Initialize environment
        self.env_config = EnvConfig_v1("GAIServiceEnv")
        self.env = GAIServiceEnv_v1(self.env_config, seed=config['seed'])
        
        # Get dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}, Max action: {self.max_action}")
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Initialize replay buffer
        if config.get('use_prioritized_replay', False):
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=config['buffer_size'],
                device=self.device
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=config['buffer_size'],
                device=self.device
            )
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.training_metrics = defaultdict(list)
        
        # Create directories
        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Initialize logging
        if config.get('use_wandb', False):
            wandb.init(
                project=config['project_name'],
                name=f"{config['agent_type']}_{config['seed']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config
            )
    
    def _create_agent(self):
        """Create agent based on configuration"""
        agent_type = self.config['agent_type'].lower()
        
        if agent_type == 'a2c_diffusion':
            return Diffusion_A2C(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                gamma=self.config.get('gamma', 0.99),
                lr=self.config.get('lr', 1e-4),
                beta_schedule=self.config.get('beta_schedule', 'linear'),
                n_timesteps=self.config.get('n_timesteps', 20),
                ema_decay=self.config.get('ema_decay', 0.995),
                step_start_ema=self.config.get('step_start_ema', 2000),
                update_ema_every=self.config.get('update_ema_every', 10),
                grad_norm=self.config.get('grad_norm', 0.25),
                entropy_coef=self.config.get('entropy_coef', 0.01),
                value_loss_coef=self.config.get('value_loss_coef', 0.25),
                lr_decay=self.config.get('lr_decay', False),
                max_grad_norm=self.config.get('max_grad_norm', 0.5)
            )
        
        elif agent_type == 'bc_diffusion':
            return Diffusion_BC(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                discount=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.005),
                beta_schedule=self.config.get('beta_schedule', 'linear'),
                n_timesteps=self.config.get('n_timesteps', 100),
                lr=self.config.get('lr', 2e-4)
            )
        
        elif agent_type == 'gaussian_a2c':
            return Gaussian_A2C(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                lr=self.config.get('lr', 3e-4),
                gamma=self.config.get('gamma', 0.99),
                value_coef=self.config.get('value_coef', 0.5),
                ent_coef=self.config.get('ent_coef', 0.01),
                grad_norm=self.config.get('grad_norm', 1.0)
            )
        
        elif agent_type == 'gaussian_dql':
            return Gaussian_DQL(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                discount=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.005),
                lr=self.config.get('lr', 3e-4),
                grad_norm=self.config.get('grad_norm', 1.0),
                noise_scale=self.config.get('noise_scale', 0.3),
                noise_type=self.config.get('noise_type', 'gaussian'),
                epsilon=self.config.get('epsilon', 0.01),
                policy_delay=self.config.get('policy_delay', 2)
            )
        
        elif agent_type == 'gaussian_ppo':
            return Gaussian_PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                discount=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.005),
                lr=self.config.get('lr', 7e-3),
                lr_decay=self.config.get('lr_decay', False),
                lr_maxt=self.config.get('lr_maxt', 1000),
                grad_norm=self.config.get('grad_norm', 1.0),
                clip_ratio=self.config.get('clip_ratio', 0.2),
                value_clip_ratio=self.config.get('value_clip_ratio', 0.2),
                norm_adv=self.config.get('norm_adv', True),
                horizon_steps=self.config.get('horizon_steps', 1),
                ent_coef=self.config.get('ent_coef', 0.01),
                noise_scale=self.config.get('noise_scale', 0.1),
                noise_type=self.config.get('noise_type', 'gaussian'),
                epsilon=self.config.get('epsilon', 0.1)
            )
        
        elif agent_type == 'ppo_diffusion':
            return Diffusion_PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                gamma=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.95),
                clip_param=self.config.get('clip_param', 0.2),
                beta_schedule=self.config.get('beta_schedule', 'linear'),
                n_timesteps=self.config.get('n_timesteps', 10),
                ema_decay=self.config.get('ema_decay', 0.99),
                step_start_ema=self.config.get('step_start_ema', 500),
                update_ema_every=self.config.get('update_ema_every', 3),
                lr=self.config.get('lr', 2e-4),
                lr_decay=self.config.get('lr_decay', True),
                lr_maxt=self.config.get('lr_maxt', 10000),
                grad_norm=self.config.get('grad_norm', 1.0),
                entropy_coef=self.config.get('entropy_coef', 0.02),
                value_loss_coef=self.config.get('value_loss_coef', 0.25),
                warmup_steps=self.config.get('warmup_steps', 200)
            )
        
        elif agent_type == 'ql_diffusion':
            return Diffusion_QL(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                discount=self.config.get('gamma', 0.99),
                tau=self.config.get('tau', 0.005),
                max_q_backup=self.config.get('max_q_backup', False),
                eta=self.config.get('eta', 1.0),
                beta_schedule=self.config.get('beta_schedule', 'linear'),
                n_timesteps=self.config.get('n_timesteps', 100),
                ema_decay=self.config.get('ema_decay', 0.995),
                step_start_ema=self.config.get('step_start_ema', 1000),
                update_ema_every=self.config.get('update_ema_every', 5),
                lr=self.config.get('lr', 3e-4),
                lr_decay=self.config.get('lr_decay', False),
                lr_maxt=self.config.get('lr_maxt', 1000),
                grad_norm=self.config.get('grad_norm', 1.0)
            )
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def train(self):
        """Main training loop"""
        print(f"Starting training with {self.config['agent_type']} agent...")
        print(f"Total episodes: {self.config['total_episodes']}")
        print(f"Max steps per episode: {self.config['max_steps_per_episode']}")
        
        start_time = time.time()
        
        for episode in range(self.config['total_episodes']):
            episode_reward, episode_length, episode_info = self._run_episode(episode)
            
            # Store metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Training step
            if len(self.replay_buffer) >= self.config['min_buffer_size']:
                metrics = self.agent.train(
                    self.replay_buffer,
                    iterations=self.config['train_iterations'],
                    batch_size=self.config['batch_size']
                )
                
                # Store training metrics
                for key, values in metrics.items():
                    self.training_metrics[key].extend(values)
            
            # Logging
            if episode % self.config['log_interval'] == 0:
                self._log_metrics(episode, episode_reward, episode_length, episode_info)
            
            # Save model
            if episode % self.config['save_interval'] == 0 and episode > 0:
                self._save_model(episode)
            
            # Evaluation
            if episode % self.config['eval_interval'] == 0 and episode > 0:
                self._evaluate(episode)
        
        # Final save
        self._save_model('final')
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    def _run_episode(self, episode):
        """Run a single episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_info = {}
        
        for step in range(self.config['max_steps_per_episode']):
            # Select action
            if episode < self.config.get('explore_episodes', 100):
                # Exploration phase
                action = self.env.action_space.sample()
            else:
                # Use agent policy
                action = self.agent.sample_action(state)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.replay_buffer.push(state, action, next_state, reward, done)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Store episode info
            if step == 0:  # Store info from first step
                episode_info = info
            
            if done:
                break
        
        return episode_reward, episode_length, episode_info
    
    def _log_metrics(self, episode, episode_reward, episode_length, episode_info):
        """Log training metrics"""
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        
        print(f"Episode {episode:6d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Length: {episode_length:3d} | "
              f"Avg Length: {avg_length:5.1f}")
        
        if episode_info:
            print(f"  Served: {episode_info.get('total_served', 0)} | "
                  f"Latency: {episode_info.get('total_latency', 0):.4f} | "
                  f"FLOPS: {episode_info.get('total_flops', 0):.0f} | "
                  f"Memory: {episode_info.get('total_mem', 0):.3f} | "
                  f"Penalty: {episode_info.get('penalty', 0):.2f} | "
                  f"Bonus: {episode_info.get('bonus', 0):.2f}")
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            log_dict = {
                'episode': episode,
                'episode_reward': episode_reward,
                'avg_reward': avg_reward,
                'episode_length': episode_length,
                'avg_length': avg_length,
                'buffer_size': len(self.replay_buffer)
            }
            
            if episode_info:
                log_dict.update({
                    'served_users': episode_info.get('total_served', 0),
                    'total_latency': episode_info.get('total_latency', 0),
                    'total_flops': episode_info.get('total_flops', 0),
                    'total_memory': episode_info.get('total_mem', 0),
                    'penalty': episode_info.get('penalty', 0),
                    'bonus': episode_info.get('bonus', 0)
                })
            
            wandb.log(log_dict)
    
    def _evaluate(self, episode):
        """Evaluate the agent"""
        print(f"\nEvaluating at episode {episode}...")
        
        eval_episodes = self.config.get('eval_episodes', 5)
        eval_rewards = []
        eval_lengths = []
        
        for _ in range(eval_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for _ in range(self.config['max_steps_per_episode']):
                action = self.agent.sample_action(state, deterministic=True)
                next_state, reward, done, _ = self.env.step(action)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
        
        avg_eval_reward = np.mean(eval_rewards)
        avg_eval_length = np.mean(eval_lengths)
        
        print(f"Evaluation - Avg Reward: {avg_eval_reward:.2f}, Avg Length: {avg_eval_length:.1f}")
        
        if self.config.get('use_wandb', False):
            wandb.log({
                'eval_avg_reward': avg_eval_reward,
                'eval_avg_length': avg_eval_length,
                'eval_episode': episode
            })
    
    def _save_model(self, episode):
        """Save the model"""
        save_path = os.path.join(self.config['save_dir'], f'model_episode_{episode}')
        self.agent.save_model(save_path)
        print(f"Model saved to {save_path}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        if not self.training_metrics:
            print("No training metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Episode lengths
        axes[0, 1].plot(self.episode_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        
        # Training losses
        if 'actor_loss' in self.training_metrics:
            axes[1, 0].plot(self.training_metrics['actor_loss'])
            axes[1, 0].set_title('Actor Loss')
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Loss')
        
        if 'critic_loss' in self.training_metrics:
            axes[1, 1].plot(self.training_metrics['critic_loss'])
            axes[1, 1].set_title('Critic Loss')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['log_dir'], 'training_curves.png'))
        plt.show()


def get_default_config():
    """Get default configuration"""
    return {
        # Environment
        'env_name': 'GAIServiceEnv',
        'seed': 42,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # Agent
        'agent_type': 'gaussian_a2c',  # Options: a2c_diffusion, bc_diffusion, gaussian_a2c, gaussian_dql, gaussian_ppo, ppo_diffusion, ql_diffusion
        
        # Training
        'total_episodes': 1000,
        'max_steps_per_episode': 10,
        'explore_episodes': 100,
        'min_buffer_size': 1000,
        'train_iterations': 10,
        'batch_size': 64,
        
        # Replay Buffer
        'buffer_size': 100000,
        'use_prioritized_replay': False,
        
        # Logging
        'log_interval': 10,
        'save_interval': 100,
        'eval_interval': 50,
        'eval_episodes': 5,
        'use_wandb': False,
        'project_name': 'drl-memory-maximize',
        
        # Directories
        'save_dir': 'models',
        'log_dir': 'logs',
        
        # Hyperparameters (will be overridden by agent-specific defaults)
        'gamma': 0.99,
        'lr': 3e-4,
        'tau': 0.005,
        'grad_norm': 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description='Train DRL agents on Memory Maximization Problem')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--agent', type=str, choices=[
        'a2c_diffusion', 'bc_diffusion', 'gaussian_a2c', 
        'gaussian_dql', 'gaussian_ppo', 'ppo_diffusion', 'ql_diffusion'
    ], help='Agent type')
    parser.add_argument('--episodes', type=int, help='Number of episodes')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    
    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Override with command line arguments
    if args.agent:
        config['agent_type'] = args.agent
    if args.episodes:
        config['total_episodes'] = args.episodes
    if args.lr:
        config['lr'] = args.lr
    if args.seed:
        config['seed'] = args.seed
    if args.device:
        config['device'] = args.device
    if args.wandb:
        config['use_wandb'] = True
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()
    trainer.plot_training_curves()


if __name__ == '__main__':
    main()