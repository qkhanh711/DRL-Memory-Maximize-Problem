"""
Comprehensive metrics plotting script for DRL training analysis.
Plots convergence curves, performance metrics, and comparative analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib defaults
plt.style.use("default")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = 'gray'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Color palette
COLORS = {
    'gaussian_ppo': '#1f77b4',
    'gaussian_dql': '#ff7f0e',
    'ppo_diffusion': '#2ca02c',
    'ql_diffusion': '#d62728'
}

ALGORITHM_NAMES = {
    'gaussian_ppo': 'PPO',
    'gaussian_dql': 'DQL',
    'ppo_diffusion': 'DiffPPO',
    'ql_diffusion': 'DiffQL'
}


def load_data(filepath):
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def plot_convergence_curves(data, save_path='plots/convergence.png'):
    """Plot convergence curves for all algorithms."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for alg, alg_data in data.items():
        episodes = alg_data['episodes']
        rewards = alg_data['rewards']
        
        # Plot with smooth rolling average
        window = max(10, len(rewards) // 20)
        rolling_avg = pd.Series(rewards).rolling(window=window, center=True).mean()
        
        label = ALGORITHM_NAMES.get(alg, alg)
        ax.plot(episodes, rolling_avg, linewidth=2.5, label=label, 
                color=COLORS.get(alg, None), alpha=0.8)
        ax.fill_between(episodes, rolling_avg - np.std(rewards)/5, 
                        rolling_avg + np.std(rewards)/5, 
                        alpha=0.15, color=COLORS.get(alg, None))
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title('Training Convergence Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_reward_distribution(data, save_path='plots/reward_distribution.png'):
    """Plot reward distribution as box plots and violin plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prepare data
    rewards_data = []
    labels = []
    for alg, alg_data in data.items():
        rewards_data.append(alg_data['rewards'])
        labels.append(ALGORITHM_NAMES.get(alg, alg))
    
    # Box plot
    bp = axes[0].boxplot(rewards_data, labels=labels, patch_artist=True)
    for patch, alg in zip(bp['boxes'], data.keys()):
        patch.set_facecolor(COLORS.get(alg, '#808080'))
        patch.set_alpha(0.7)
    axes[0].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[0].set_title('Reward Distribution (Box Plot)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Violin plot
    parts = axes[1].violinplot(rewards_data, positions=range(len(rewards_data)), 
                                showmeans=True, showmedians=True)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel('Reward', fontsize=11, fontweight='bold')
    axes[1].set_title('Reward Distribution (Violin Plot)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_statistics_comparison(data, save_path='plots/statistics_comparison.png'):
    """Plot statistical comparison of algorithms."""
    stats_list = []
    
    for alg, alg_data in data.items():
        rewards = np.array(alg_data['rewards'])
        stats_list.append({
            'Algorithm': ALGORITHM_NAMES.get(alg, alg),
            'Mean': np.mean(rewards),
            'Std': np.std(rewards),
            'Min': np.min(rewards),
            'Max': np.max(rewards),
            'Median': np.median(rewards)
        })
    
    df = pd.DataFrame(stats_list)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    metrics = ['Mean', 'Std', 'Min', 'Max', 'Median']
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(df['Algorithm'], df[metric], 
                      color=[COLORS.get(list(data.keys())[i], '#808080') 
                             for i in range(len(data))])
        ax.set_title(f'{metric} Reward', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Remove empty subplot
    axes[-1].axis('off')
    
    # Add statistics table
    table_data = df.set_index('Algorithm').round(2).values
    table = axes[-1].table(cellText=table_data, 
                          colLabels=df.set_index('Algorithm').columns,
                          rowLabels=df['Algorithm'],
                          cellLoc='center', loc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_convergence_analysis(data, save_path='plots/convergence_analysis.png'):
    """Plot convergence analysis: learning speed and stability."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    learning_speeds = []
    stabilities = []
    alg_names = []
    
    for alg, alg_data in data.items():
        rewards = np.array(alg_data['rewards'])
        episodes = np.array(alg_data['episodes'])
        
        # Learning speed: improvement from first to last 100 episodes
        first_100 = np.mean(rewards[:min(100, len(rewards))])
        last_100 = np.mean(rewards[-min(100, len(rewards)):])
        learning_speed = ((last_100 - first_100) / (abs(first_100) + 1e-6)) * 100
        
        # Stability: inverse of coefficient of variation in last 100 episodes
        last_100_rewards = rewards[-min(100, len(rewards)):]
        stability = 1.0 / (np.std(last_100_rewards) / (np.mean(last_100_rewards) + 1e-6) + 1e-6)
        
        learning_speeds.append(learning_speed)
        stabilities.append(stability)
        alg_names.append(ALGORITHM_NAMES.get(alg, alg))
    
    # Learning speed
    bars1 = axes[0].bar(alg_names, learning_speeds,
                       color=[COLORS.get(alg, '#808080') for alg in data.keys()])
    axes[0].set_title('Learning Speed (% Improvement)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Improvement %', fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    # Stability
    bars2 = axes[1].bar(alg_names, stabilities,
                       color=[COLORS.get(alg, '#808080') for alg in data.keys()])
    axes[1].set_title('Training Stability', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Stability Score', fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def plot_reward_trajectory(data, save_path='plots/reward_trajectory.png'):
    """Plot individual reward trajectories with trend lines."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (alg, alg_data) in enumerate(data.items()):
        ax = axes[idx]
        episodes = np.array(alg_data['episodes'])
        rewards = np.array(alg_data['rewards'])
        
        # Raw rewards
        ax.plot(episodes, rewards, '.', alpha=0.3, markersize=4, 
                color=COLORS.get(alg, '#808080'), label='Raw rewards')
        
        # Trend line using moving average
        window = max(20, len(rewards) // 15)
        trend = pd.Series(rewards).rolling(window=window, center=True).mean()
        ax.plot(episodes, trend, linewidth=2.5, 
                color=COLORS.get(alg, '#808080'), label='Trend (MA)')
        
        # Polynomial fit
        z = np.polyfit(episodes, rewards, 3)
        p = np.poly1d(z)
        ax.plot(episodes, p(episodes), '--', linewidth=2, 
                color='red', alpha=0.7, label='Polynomial fit')
        
        ax.set_title(f'{ALGORITHM_NAMES.get(alg, alg)} - Reward Trajectory', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=10)
        ax.set_ylabel('Reward', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()


def create_summary_report(data):
    """Generate a text summary report of metrics."""
    print("\n" + "="*80)
    print("TRAINING METRICS SUMMARY REPORT")
    print("="*80 + "\n")
    
    for alg, alg_data in data.items():
        rewards = np.array(alg_data['rewards'])
        print(f"\n📊 {ALGORITHM_NAMES.get(alg, alg).upper()}")
        print("-" * 40)
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Std Dev: {np.std(rewards):.2f}")
        print(f"  Min Reward: {np.min(rewards):.2f}")
        print(f"  Max Reward: {np.max(rewards):.2f}")
        print(f"  Median: {np.median(rewards):.2f}")
        print(f"  Q1 (25%): {np.percentile(rewards, 25):.2f}")
        print(f"  Q3 (75%): {np.percentile(rewards, 75):.2f}")
        print(f"  IQR: {np.percentile(rewards, 75) - np.percentile(rewards, 25):.2f}")
        
        # Learning progress
        first_100 = np.mean(rewards[:min(100, len(rewards))])
        last_100 = np.mean(rewards[-min(100, len(rewards)):])
        improvement = ((last_100 - first_100) / (abs(first_100) + 1e-6)) * 100
        print(f"  First 100 avg: {first_100:.2f}")
        print(f"  Last 100 avg: {last_100:.2f}")
        print(f"  Improvement: {improvement:.2f}%")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main function to generate all plots."""
    import os
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load convergence data
    print("📂 Loading convergence data...")
    convergence_data = load_data('backup_saved_plots/convergence_data.json')
    
    if convergence_data is None:
        print("❌ Could not load convergence data!")
        return
    
    print(f"✅ Loaded data for {len(convergence_data)} algorithms\n")
    
    # Generate plots
    print("📈 Generating plots...\n")
    
    print("1️⃣  Convergence Curves...")
    plot_convergence_curves(convergence_data, 'plots/01_convergence_curves.png')
    
    print("2️⃣  Reward Distribution...")
    plot_reward_distribution(convergence_data, 'plots/02_reward_distribution.png')
    
    print("3️⃣  Statistics Comparison...")
    plot_statistics_comparison(convergence_data, 'plots/03_statistics_comparison.png')
    
    print("4️⃣  Convergence Analysis...")
    plot_convergence_analysis(convergence_data, 'plots/04_convergence_analysis.png')
    
    print("5️⃣  Reward Trajectories...")
    plot_reward_trajectory(convergence_data, 'plots/05_reward_trajectories.png')
    
    # Generate summary report
    create_summary_report(convergence_data)
    
    print("✨ All plots generated successfully!")
    print("📁 Plots saved to: ./plots/")


if __name__ == '__main__':
    main()
