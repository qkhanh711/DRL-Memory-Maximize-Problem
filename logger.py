import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob
import argparse

# Try to import seaborn, but don't require it
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

class MetricsLogger:
    """Logger for reading and visualizing training metrics"""
    
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.metrics = {}
        
    def load_metrics(self, log_dir):
        """Load metrics from a log directory"""
        self.log_dir = log_dir
        metrics_file = os.path.join(log_dir, 'training_metrics.json')
        
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)
        else:
            print(f"No metrics file found at {metrics_file}")
            
    def load_multiple_logs(self, base_dir='logs'):
        """Load metrics from multiple log directories"""
        self.metrics = {}
        
        # Find all log directories
        log_dirs = glob.glob(os.path.join(base_dir, '*'))
        
        for log_dir in log_dirs:
            if os.path.isdir(log_dir):
                metrics_file = os.path.join(log_dir, 'training_metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        data = json.load(f)
                        agent_name = data.get('agent_name', 'unknown')
                        self.metrics[agent_name] = data
                        
    def plot_training_curves(self, save_path='training_curves.png', figsize=(15, 10)):
        """Plot training curves for all agents"""
        if not self.metrics:
            print("No metrics loaded. Call load_metrics() or load_multiple_logs() first.")
            return
            
        # Set up the plot style
        if HAS_SEABORN:
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
        else:
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Training Progress Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        for agent_name, data in self.metrics.items():
            if 'episode_rewards' in data:
                rewards = data['episode_rewards']
                # Smooth the curve using moving average
                window_size = min(50, len(rewards) // 10)
                if window_size > 1:
                    smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                    episodes = np.arange(len(smoothed_rewards))
                else:
                    smoothed_rewards = rewards
                    episodes = np.arange(len(rewards))
                ax1.plot(episodes, smoothed_rewards, label=agent_name, alpha=0.8, linewidth=2)
        
        ax1.set_title('Episode Rewards (Smoothed)', fontweight='bold')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        for agent_name, data in self.metrics.items():
            if 'episode_lengths' in data:
                lengths = data['episode_lengths']
                window_size = min(50, len(lengths) // 10)
                if window_size > 1:
                    smoothed_lengths = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
                    episodes = np.arange(len(smoothed_lengths))
                else:
                    smoothed_lengths = lengths
                    episodes = np.arange(len(lengths))
                ax2.plot(episodes, smoothed_lengths, label=agent_name, alpha=0.8, linewidth=2)
        
        ax2.set_title('Episode Lengths (Smoothed)', fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Evaluation Rewards
        ax3 = axes[1, 0]
        for agent_name, data in self.metrics.items():
            if 'eval_rewards' in data and data['eval_rewards']:
                eval_rewards = data['eval_rewards']
                eval_episodes = np.arange(len(eval_rewards)) * 50  # Assuming eval every 50 episodes
                ax3.plot(eval_episodes, eval_rewards, label=agent_name, marker='o', alpha=0.8, linewidth=2)
        
        ax3.set_title('Evaluation Rewards', fontweight='bold')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Eval Reward')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final Performance Comparison
        ax4 = axes[1, 1]
        agent_names = []
        final_rewards = []
        
        for agent_name, data in self.metrics.items():
            if 'episode_rewards' in data and data['episode_rewards']:
                rewards = data['episode_rewards']
                # Use last 100 episodes for final performance
                final_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                agent_names.append(agent_name)
                final_rewards.append(final_reward)
        
        if agent_names:
            bars = ax4.bar(agent_names, final_rewards, alpha=0.7)
            ax4.set_title('Final Performance (Last 100 Episodes)', fontweight='bold')
            ax4.set_xlabel('Agent')
            ax4.set_ylabel('Average Reward')
            ax4.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, final_rewards):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to: {save_path}")
        
    def plot_individual_agent(self, agent_name, save_path=None, figsize=(12, 8)):
        """Plot detailed metrics for a single agent"""
        if agent_name not in self.metrics:
            print(f"Agent {agent_name} not found in loaded metrics")
            return
            
        data = self.metrics[agent_name]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'{agent_name} - Detailed Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Episode Rewards
        ax1 = axes[0, 0]
        if 'episode_rewards' in data:
            rewards = data['episode_rewards']
            ax1.plot(rewards, alpha=0.6, color='blue', linewidth=1)
            
            # Add moving average
            window_size = min(50, len(rewards) // 10)
            if window_size > 1:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                ax1.plot(range(window_size-1, len(rewards)), smoothed, color='red', linewidth=2, label=f'MA({window_size})')
                ax1.legend()
            
        ax1.set_title('Episode Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Episode Lengths
        ax2 = axes[0, 1]
        if 'episode_lengths' in data:
            lengths = data['episode_lengths']
            ax2.plot(lengths, alpha=0.6, color='green', linewidth=1)
            
            # Add moving average
            if window_size > 1:
                smoothed = np.convolve(lengths, np.ones(window_size)/window_size, mode='valid')
                ax2.plot(range(window_size-1, len(lengths)), smoothed, color='red', linewidth=2, label=f'MA({window_size})')
                ax2.legend()
                
        ax2.set_title('Episode Lengths')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Length')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Evaluation Rewards
        ax3 = axes[1, 0]
        if 'eval_rewards' in data and data['eval_rewards']:
            eval_rewards = data['eval_rewards']
            eval_episodes = np.arange(len(eval_rewards)) * 50  # Assuming eval every 50 episodes
            ax3.plot(eval_episodes, eval_rewards, marker='o', color='purple', linewidth=2)
            
        ax3.set_title('Evaluation Rewards')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Eval Reward')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Reward Distribution
        ax4 = axes[1, 1]
        if 'episode_rewards' in data:
            rewards = data['episode_rewards']
            ax4.hist(rewards, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax4.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
            ax4.axvline(np.median(rewards), color='blue', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
            ax4.legend()
            
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Reward')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{agent_name}_detailed_metrics.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Detailed metrics for {agent_name} saved to: {save_path}")
        
    def print_summary(self):
        """Print a summary of all loaded metrics"""
        if not self.metrics:
            print("No metrics loaded.")
            return
            
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for agent_name, data in self.metrics.items():
            print(f"\n{agent_name.upper()}:")
            print("-" * 40)
            
            if 'episode_rewards' in data and data['episode_rewards']:
                rewards = data['episode_rewards']
                print(f"  Total Episodes: {len(rewards)}")
                print(f"  Final Reward (last 100): {np.mean(rewards[-100:]):.2f}")
                print(f"  Best Reward: {np.max(rewards):.2f}")
                print(f"  Mean Reward: {np.mean(rewards):.2f}")
                print(f"  Std Reward: {np.std(rewards):.2f}")
                
            if 'episode_lengths' in data and data['episode_lengths']:
                lengths = data['episode_lengths']
                print(f"  Mean Length: {np.mean(lengths):.2f}")
                print(f"  Std Length: {np.std(lengths):.2f}")
                
            if 'eval_rewards' in data and data['eval_rewards']:
                eval_rewards = data['eval_rewards']
                print(f"  Final Eval Reward: {eval_rewards[-1]:.2f}")
                print(f"  Best Eval Reward: {np.max(eval_rewards):.2f}")
                print(f"  Mean Eval Reward: {np.mean(eval_rewards):.2f}")
                
    def save_comparison_table(self, save_path='comparison_table.csv'):
        """Save a comparison table of all agents"""
        if not self.metrics:
            print("No metrics loaded.")
            return
            
        import pandas as pd
        
        comparison_data = []
        
        for agent_name, data in self.metrics.items():
            row = {'Agent': agent_name}
            
            if 'episode_rewards' in data and data['episode_rewards']:
                rewards = data['episode_rewards']
                row['Total Episodes'] = len(rewards)
                row['Final Reward (last 100)'] = np.mean(rewards[-100:])
                row['Best Reward'] = np.max(rewards)
                row['Mean Reward'] = np.mean(rewards)
                row['Std Reward'] = np.std(rewards)
                
            if 'episode_lengths' in data and data['episode_lengths']:
                lengths = data['episode_lengths']
                row['Mean Length'] = np.mean(lengths)
                row['Std Length'] = np.std(lengths)
                
            if 'eval_rewards' in data and data['eval_rewards']:
                eval_rewards = data['eval_rewards']
                row['Final Eval Reward'] = eval_rewards[-1]
                row['Best Eval Reward'] = np.max(eval_rewards)
                row['Mean Eval Reward'] = np.mean(eval_rewards)
                
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df.to_csv(save_path, index=False)
        print(f"Comparison table saved to: {save_path}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory containing log files')
    parser.add_argument('--agent', type=str, default=None, help='Specific agent to plot (optional)')
    parser.add_argument('--save_dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize logger
    logger = MetricsLogger()
    
    # Load metrics
    logger.load_multiple_logs(args.log_dir)
    
    if not logger.metrics:
        print(f"No metrics found in {args.log_dir}")
        return
    
    # Print summary
    logger.print_summary()
    
    # Generate plots
    if args.agent:
        # Plot specific agent
        logger.plot_individual_agent(args.agent, 
                                   save_path=os.path.join(args.save_dir, f'{args.agent}_detailed.png'))
    else:
        # Plot all agents
        logger.plot_training_curves(save_path=os.path.join(args.save_dir, 'training_curves.png'))
        
        # Plot individual agents
        for agent_name in logger.metrics.keys():
            logger.plot_individual_agent(agent_name, 
                                       save_path=os.path.join(args.save_dir, f'{agent_name}_detailed.png'))
    
    # Save comparison table
    logger.save_comparison_table(save_path=os.path.join(args.save_dir, 'comparison_table.csv'))

if __name__ == "__main__":
    main()