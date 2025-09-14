#!/usr/bin/env python3
"""
Evaluation script for trained DRL agents
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
from agents.a2c_diffusion import Diffusion_A2C
from agents.bc_diffusion import Diffusion_BC
from agents.gaussian_a2c import Gaussian_A2C
from agents.gaussian_dql import Gaussian_DQL
from agents.gaussian_ppo import Gaussian_PPO
from agents.ppo_diffusion import Diffusion_PPO
from agents.ql_diffusion import Diffusion_QL


class Evaluator:
    """Evaluator for trained agents"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Initialize environment
        self.env_config = EnvConfig_v1("GAIServiceEnv")
        self.env = GAIServiceEnv_v1(self.env_config, seed=config['seed'])
        
        # Get dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])
        
        # Load agent
        self.agent = self._load_agent()
        
    def _load_agent(self):
        """Load trained agent"""
        agent_type = self.config['agent_type'].lower()
        model_path = self.config['model_path']
        
        if agent_type == 'a2c_diffusion':
            agent = Diffusion_A2C(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        elif agent_type == 'bc_diffusion':
            agent = Diffusion_BC(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        elif agent_type == 'gaussian_a2c':
            agent = Gaussian_A2C(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        elif agent_type == 'gaussian_dql':
            agent = Gaussian_DQL(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        elif agent_type == 'gaussian_ppo':
            agent = Gaussian_PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        elif agent_type == 'ppo_diffusion':
            agent = Diffusion_PPO(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        elif agent_type == 'ql_diffusion':
            agent = Diffusion_QL(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                max_action=self.max_action,
                device=self.device,
                **self.config.get('agent_params', {})
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Load model weights
        agent.load_model(model_path)
        return agent
    
    def evaluate(self, num_episodes=100, deterministic=True):
        """Evaluate the agent"""
        print(f"Evaluating {self.config['agent_type']} agent for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        episode_metrics = defaultdict(list)
        
        for episode in range(num_episodes):
            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}")
            
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            episode_info = {}
            
            for step in range(self.config.get('max_steps_per_episode', 10)):
                # Select action
                action = self.agent.sample_action(state, deterministic=deterministic)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Update metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Store episode info from first step
                if step == 0:
                    episode_info = info
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Store detailed metrics
            if episode_info:
                for key, value in episode_info.items():
                    if isinstance(value, (int, float)):
                        episode_metrics[key].append(value)
        
        # Compute statistics
        stats = self._compute_statistics(episode_rewards, episode_lengths, episode_metrics)
        
        return stats, episode_rewards, episode_lengths, episode_metrics
    
    def _compute_statistics(self, rewards, lengths, metrics):
        """Compute evaluation statistics"""
        stats = {
            'rewards': {
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'median': np.median(rewards)
            },
            'lengths': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': np.min(lengths),
                'max': np.max(lengths),
                'median': np.median(lengths)
            }
        }
        
        # Add metric statistics
        for key, values in metrics.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    def compare_agents(self, agent_configs, num_episodes=50):
        """Compare multiple agents"""
        print(f"Comparing {len(agent_configs)} agents for {num_episodes} episodes each...")
        
        results = {}
        
        for i, agent_config in enumerate(agent_configs):
            print(f"\nEvaluating agent {i+1}/{len(agent_configs)}: {agent_config['agent_type']}")
            
            # Update config
            self.config.update(agent_config)
            
            # Load new agent
            self.agent = self._load_agent()
            
            # Evaluate
            stats, rewards, lengths, metrics = self.evaluate(num_episodes)
            
            results[agent_config['agent_type']] = {
                'stats': stats,
                'rewards': rewards,
                'lengths': lengths,
                'metrics': metrics
            }
        
        return results
    
    def plot_results(self, stats, rewards, lengths, metrics, save_path=None):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Episode rewards
        axes[0, 0].hist(rewards, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['rewards']['mean'], color='red', linestyle='--', 
                          label=f"Mean: {stats['rewards']['mean']:.2f}")
        axes[0, 0].set_title('Episode Rewards Distribution')
        axes[0, 0].set_xlabel('Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[0, 1].hist(lengths, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(stats['lengths']['mean'], color='red', linestyle='--',
                          label=f"Mean: {stats['lengths']['mean']:.2f}")
        axes[0, 1].set_title('Episode Lengths Distribution')
        axes[0, 1].set_xlabel('Length')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Reward over episodes
        axes[0, 2].plot(rewards, alpha=0.7)
        axes[0, 2].axhline(stats['rewards']['mean'], color='red', linestyle='--',
                          label=f"Mean: {stats['rewards']['mean']:.2f}")
        axes[0, 2].set_title('Rewards Over Episodes')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Reward')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Key metrics
        if 'total_served' in metrics:
            axes[1, 0].hist(metrics['total_served'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Users Served Distribution')
            axes[1, 0].set_xlabel('Users Served')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        if 'total_latency' in metrics:
            axes[1, 1].hist(metrics['total_latency'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Total Latency Distribution')
            axes[1, 1].set_xlabel('Latency (s)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        if 'total_flops' in metrics:
            axes[1, 2].hist(metrics['total_flops'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 2].set_title('Total FLOPS Distribution')
            axes[1, 2].set_xlabel('FLOPS')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('evaluation_results.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_comparison(self, results, save_path=None):
        """Plot comparison between agents"""
        agent_names = list(results.keys())
        
        # Extract metrics for comparison
        reward_means = [results[name]['stats']['rewards']['mean'] for name in agent_names]
        reward_stds = [results[name]['stats']['rewards']['std'] for name in agent_names]
        
        served_means = []
        served_stds = []
        for name in agent_names:
            if 'total_served' in results[name]['stats']:
                served_means.append(results[name]['stats']['total_served']['mean'])
                served_stds.append(results[name]['stats']['total_served']['std'])
            else:
                served_means.append(0)
                served_stds.append(0)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Reward comparison
        x_pos = np.arange(len(agent_names))
        axes[0].bar(x_pos, reward_means, yerr=reward_stds, capsize=5, alpha=0.7)
        axes[0].set_title('Average Episode Rewards')
        axes[0].set_xlabel('Agent')
        axes[0].set_ylabel('Reward')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(agent_names, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Users served comparison
        axes[1].bar(x_pos, served_means, yerr=served_stds, capsize=5, alpha=0.7)
        axes[1].set_title('Average Users Served')
        axes[1].set_xlabel('Agent')
        axes[1].set_ylabel('Users Served')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(agent_names, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig('agent_comparison.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_results(self, stats, rewards, lengths, metrics, filename='evaluation_results.json'):
        """Save evaluation results to file"""
        results = {
            'stats': stats,
            'rewards': rewards,
            'lengths': lengths,
            'metrics': metrics,
            'config': self.config
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained DRL agents')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic actions')
    parser.add_argument('--compare', type=str, nargs='+', help='Compare multiple agents')
    parser.add_argument('--save_plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config['model_path'] = args.model_path
    config['deterministic'] = args.deterministic
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    if args.compare:
        # Compare multiple agents
        agent_configs = []
        for agent_type in args.compare:
            agent_config = config.copy()
            agent_config['agent_type'] = agent_type
            agent_configs.append(agent_config)
        
        results = evaluator.compare_agents(agent_configs, args.episodes)
        
        # Print comparison results
        print("\n" + "="*50)
        print("AGENT COMPARISON RESULTS")
        print("="*50)
        
        for agent_name, result in results.items():
            print(f"\n{agent_name}:")
            print(f"  Average Reward: {result['stats']['rewards']['mean']:.2f} ± {result['stats']['rewards']['std']:.2f}")
            if 'total_served' in result['stats']:
                print(f"  Average Users Served: {result['stats']['total_served']['mean']:.2f} ± {result['stats']['total_served']['std']:.2f}")
            if 'total_latency' in result['stats']:
                print(f"  Average Latency: {result['stats']['total_latency']['mean']:.4f} ± {result['stats']['total_latency']['std']:.4f}")
        
        # Plot comparison
        evaluator.plot_comparison(results, 'agent_comparison.png' if args.save_plots else None)
        
    else:
        # Single agent evaluation
        stats, rewards, lengths, metrics = evaluator.evaluate(args.episodes, args.deterministic)
        
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Episodes: {args.episodes}")
        print(f"Average Reward: {stats['rewards']['mean']:.2f} ± {stats['rewards']['std']:.2f}")
        print(f"Average Length: {stats['lengths']['mean']:.2f} ± {stats['lengths']['std']:.2f}")
        
        if 'total_served' in stats:
            print(f"Average Users Served: {stats['total_served']['mean']:.2f} ± {stats['total_served']['std']:.2f}")
        if 'total_latency' in stats:
            print(f"Average Latency: {stats['total_latency']['mean']:.4f} ± {stats['total_latency']['std']:.4f}")
        if 'total_flops' in stats:
            print(f"Average FLOPS: {stats['total_flops']['mean']:.0f} ± {stats['total_flops']['std']:.0f}")
        if 'total_mem' in stats:
            print(f"Average Memory: {stats['total_mem']['mean']:.3f} ± {stats['total_mem']['std']:.3f}")
        
        # Plot results
        evaluator.plot_results(stats, rewards, lengths, metrics, 
                              'evaluation_results.png' if args.save_plots else None)
        
        # Save results
        evaluator.save_results(stats, rewards, lengths, metrics)


if __name__ == '__main__':
    main()