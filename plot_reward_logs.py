import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--qos_required', type=float, default=25.0, help='Required QoS level for the environment')
parser.add_argument('--num_users', type=int, default=10, help='Number of users in the environment')

args = parser.parse_args()

# Path to logs directory
logs_dir = Path("/home/khanhnq/test/DRL-Memory-Maximize-Problem/logs")

# Collect data from all training runs
training_data = {}

for folder in sorted(logs_dir.iterdir()):

    if folder.is_dir():
        metrics_file = folder / f"training_metrics_qos_{float(args.qos_required)}_users_{args.num_users}.json"
        print(f"Loading metrics from: {metrics_file}")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                if "episode_rewards" in data:
                    training_data[folder.name] = data["episode_rewards"]
                    # print(f"✓ Loaded {folder.name}: {len(data['episode_rewards'])} episodes")
                else:
                    print(f"✗ No episode_rewards in {folder.name}")
                    # pass

# Create figure with subplots
if training_data:
    fig, axes = plt.subplots(len(training_data), 1, figsize=(14, 4*len(training_data)))
    
    # If only one training run, axes is not a list
    if len(training_data) == 1:
        axes = [axes]
    
    # Plot each training run
    for idx, (name, rewards) in enumerate(training_data.items()):
        ax = axes[idx]
        
        # Plot raw rewards
        episodes = range(len(rewards))
        ax.plot(episodes, rewards, linewidth=0.8, alpha=0.7, label='Episode Reward')
        
        # Add moving average (window=100)
        window = min(100, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_episodes = range(window-1, len(rewards))
            ax.plot(moving_episodes, moving_avg, linewidth=2, color='red', label=f'Moving Avg (window={window})')
        
        # Stats
        mean_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        min_reward = np.min(rewards)
        
        ax.axhline(y=mean_reward, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_reward:.2f}')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'{name}\n(Mean: {mean_reward:.2f}, Max: {max_reward:.2f}, Min: {min_reward:.2f})')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        ax.set_xlim(0, len(rewards))
    
    plt.tight_layout()
    
    # Save figure
    output_path = logs_dir.parent / f"reward_plot_all_runs_qos_{args.qos_required}_u{args.num_users}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")
    
    # Create combined plot
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_data)))
    
    for idx, (name, rewards) in enumerate(training_data.items()):
        episodes = range(len(rewards))
        
        # Plot moving average for cleaner view
        window = min(500, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            moving_episodes = range(window-1, len(rewards))
            ax2.plot(moving_episodes, moving_avg, linewidth=2, label=name, color=colors[idx])
        else:
            ax2.plot(episodes, rewards, linewidth=1.5, label=name, color=colors[idx])
    
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Reward', fontsize=12)
    ax2.set_title('Training Rewards - All Runs (Moving Average)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    ax2.set_xlim(0, max(len(r) for r in training_data.values()))
    plt.tight_layout()
    
    # Save combined plot
    combined_path = logs_dir.parent / f"reward_plot_combined_qos_{args.qos_required}_u{args.num_users}.png"
    plt.savefig(combined_path, dpi=100, bbox_inches='tight')
    # print(f"✓ Saved combined plot to: {combined_path}")
    
    # Print statistics
    # print("\n" + "="*60)
    # print("TRAINING STATISTICS")
    # print("="*60)
    for name, rewards in training_data.items():
        # print(f"\n{name}:")
        # print(f"  Episodes: {len(rewards)}")
        # print(f"  Mean Reward: {np.mean(rewards):.2f}")
        # print(f"  Std Dev: {np.std(rewards):.2f}")
        # print(f"  Max Reward: {np.max(rewards):.2f}")
        # print(f"  Min Reward: {np.min(rewards):.2f}")
        # print(f"  Last 10 Avg: {np.mean(rewards[-10:]):.2f}")
        pass 
    plt.show()
else:
    print("❌ No training metrics found!")
