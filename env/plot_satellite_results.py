import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pathlib import Path


def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing"""
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_training_rewards(log_dir, seed=42, num_mgu=50, save_path=None):
    """Plot training rewards for all agents"""
    
    agents = ['gaussian_dql', 'gaussian_ppo', 'ppo_diffusion', 'ql_diffusion']
    agent_names = {
        'gaussian_dql': 'Gaussian DQL',
        'gaussian_ppo': 'Gaussian PPO',
        'ppo_diffusion': 'Diffusion PPO',
        'ql_diffusion': 'Diffusion QL'
    }
    
    colors = {
        'gaussian_dql': '#1f77b4',
        'gaussian_ppo': '#ff7f0e',
        'ppo_diffusion': '#2ca02c',
        'ql_diffusion': '#d62728'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Plot 1: Episode Rewards (Raw)
    ax1 = axes[0]
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            rewards = data['episode_rewards']
            ax1.plot(rewards, label=agent_names[agent], color=colors[agent], alpha=0.3)
            
            # Plot smoothed version
            if len(rewards) > 10:
                smoothed = smooth_curve(rewards, weight=0.95)
                ax1.plot(smoothed, color=colors[agent], linewidth=2)
    
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Episode Reward', fontsize=12)
    ax1.set_title('Training Episode Rewards (Smoothed)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Evaluation Rewards
    ax2 = axes[1]
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            eval_rewards = data.get('eval_rewards', [])
            if eval_rewards:
                eval_episodes = np.arange(len(eval_rewards)) * data.get('train_config', {}).get('eval_frequency', 50)
                ax2.plot(eval_episodes, eval_rewards, label=agent_names[agent], 
                        color=colors[agent], linewidth=2, marker='o', markersize=4)
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Evaluation Reward', fontsize=12)
    ax2.set_title('Evaluation Rewards', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Moving Average (last 1000 episodes)
    ax3 = axes[2]
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            rewards = data['episode_rewards']
            if len(rewards) >= 1000:
                moving_avg = np.convolve(rewards, np.ones(1000)/1000, mode='valid')
                ax3.plot(np.arange(999, len(rewards)), moving_avg, 
                        label=agent_names[agent], color=colors[agent], linewidth=2)
    
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Average Reward (1000 episodes)', fontsize=12)
    ax3.set_title('Moving Average Rewards', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: User Distribution (Satellite vs BS)
    ax4 = axes[3]
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            sat_users = data.get('avg_satellite_users', [])
            if sat_users:
                # Plot moving average
                if len(sat_users) >= 20:
                    window = min(50, len(sat_users) // 4)
                    moving_avg = np.convolve(sat_users, np.ones(window)/window, mode='valid')
                    ax4.plot(np.arange(window-1, len(sat_users)), moving_avg,
                            label=agent_names[agent], color=colors[agent], linewidth=2)
    
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Number of Satellite Users', fontsize=12)
    ax4.set_title('Satellite vs BS User Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=num_mgu/2, color='gray', linestyle='--', alpha=0.5, label='Equal Split')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_comparison_table(log_dir, seed=42, num_mgu=50):
    """Create comparison table of final performance"""
    
    agents = ['gaussian_dql', 'gaussian_ppo', 'ppo_diffusion', 'ql_diffusion']
    agent_names = {
        'gaussian_dql': 'Gaussian DQL',
        'gaussian_ppo': 'Gaussian PPO',
        'ppo_diffusion': 'Diffusion PPO',
        'ql_diffusion': 'Diffusion QL'
    }
    
    print("\n" + "="*80)
    print(f"Performance Summary (Seed={seed}, MGUs={num_mgu})")
    print("="*80)
    print(f"{'Agent':<20} {'Final Reward':<15} {'Best Eval':<15} {'Avg Sat Users':<15}")
    print("-"*80)
    
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            # Calculate final average reward (last 100 episodes)
            rewards = data['episode_rewards']
            final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            
            # Get best evaluation reward
            eval_rewards = data.get('eval_rewards', [])
            best_eval = max(eval_rewards) if eval_rewards else 0
            
            # Get average satellite users
            sat_users = data.get('avg_satellite_users', [])
            avg_sat = np.mean(sat_users[-100:]) if len(sat_users) >= 100 else np.mean(sat_users) if sat_users else 0
            
            print(f"{agent_names[agent]:<20} {final_reward:<15.2f} {best_eval:<15.2f} {avg_sat:<15.1f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot Satellite-MEC training results')
    parser.add_argument('--log_dir', type=str, default='logs/satellite',
                       help='Directory containing training logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in training')
    parser.add_argument('--num_mgu', type=int, default=50, help='Number of MGUs')
    parser.add_argument('--save', type=str, default=None, 
                       help='Path to save the plot (e.g., results.png)')
    parser.add_argument('--table_only', action='store_true',
                       help='Only print comparison table without plotting')
    
    args = parser.parse_args()
    
    # Print comparison table
    plot_comparison_table(args.log_dir, args.seed, args.num_mgu)
    
    # Plot results
    if not args.table_only:
        save_path = args.save
        if save_path is None:
            save_path = f'satellite_training_results_seed{args.seed}_mgu{args.num_mgu}.png'
        
        plot_training_rewards(args.log_dir, args.seed, args.num_mgu, save_path)


if __name__ == "__main__":
    main()
