"""
Logging utilities for DRL training
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import torch


class Logger:
    """Simple logger for training metrics"""
    
    def __init__(self, log_dir, use_tensorboard=False, use_wandb=False):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logging backends
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir)
            except ImportError:
                print("TensorBoard not available, disabling...")
                self.use_tensorboard = False
        
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("wandb not available, disabling...")
                self.use_wandb = False
        
        # Store metrics
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        
    def log_scalar(self, key, value, step=None):
        """Log a scalar value"""
        if step is None:
            step = len(self.metrics[key])
        
        self.metrics[key].append((step, value))
        
        if self.use_tensorboard:
            self.tb_writer.add_scalar(key, value, step)
        
        if self.use_wandb:
            self.wandb.log({key: value}, step=step)
    
    def log_episode(self, episode, metrics_dict):
        """Log episode-level metrics"""
        for key, value in metrics_dict.items():
            self.episode_metrics[key].append((episode, value))
            self.log_scalar(f"episode/{key}", value, episode)
    
    def log_training_step(self, step, metrics_dict):
        """Log training step metrics"""
        for key, value in metrics_dict.items():
            self.log_scalar(f"training/{key}", value, step)
    
    def log_histogram(self, key, values, step=None):
        """Log histogram of values"""
        if step is None:
            step = len(self.metrics[key])
        
        if self.use_tensorboard:
            self.tb_writer.add_histogram(key, values, step)
        
        if self.use_wandb:
            self.wandb.log({key: self.wandb.Histogram(values)}, step=step)
    
    def log_figure(self, key, figure, step=None):
        """Log a matplotlib figure"""
        if step is None:
            step = len(self.metrics[key])
        
        if self.use_tensorboard:
            self.tb_writer.add_figure(key, figure, step)
        
        if self.use_wandb:
            self.wandb.log({key: self.wandb.Image(figure)}, step=step)
    
    def save_metrics(self, filename="metrics.json"):
        """Save all metrics to JSON file"""
        metrics_data = {
            'scalar_metrics': dict(self.metrics),
            'episode_metrics': dict(self.episode_metrics)
        }
        
        with open(os.path.join(self.log_dir, filename), 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        if not self.metrics:
            print("No metrics to plot")
            return
        
        # Create subplots
        n_metrics = len(self.metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each metric
        for i, (key, values) in enumerate(self.metrics.items()):
            if values:
                steps, vals = zip(*values)
                row, col = i // n_cols, i % n_cols
                axes[row, col].plot(steps, vals)
                axes[row, col].set_title(key)
                axes[row, col].set_xlabel('Step')
                axes[row, col].set_ylabel('Value')
                axes[row, col].grid(True)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def close(self):
        """Close the logger"""
        if self.use_tensorboard:
            self.tb_writer.close()
        
        if self.use_wandb:
            self.wandb.finish()


class MetricsTracker:
    """Track and compute various metrics during training"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.training_losses = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_episode(self, reward, length, info=None):
        """Add episode data"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.training_losses[key].append(value)
    
    def get_stats(self):
        """Get current statistics"""
        stats = {}
        
        if self.episode_rewards:
            stats['avg_reward'] = np.mean(self.episode_rewards)
            stats['std_reward'] = np.std(self.episode_rewards)
            stats['min_reward'] = np.min(self.episode_rewards)
            stats['max_reward'] = np.max(self.episode_rewards)
        
        if self.episode_lengths:
            stats['avg_length'] = np.mean(self.episode_lengths)
            stats['std_length'] = np.std(self.episode_lengths)
        
        for key, values in self.training_losses.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
        
        return stats
    
    def get_recent_trend(self, metric='reward', window=10):
        """Get recent trend of a metric"""
        if metric == 'reward' and self.episode_rewards:
            recent = list(self.episode_rewards)[-window:]
            if len(recent) >= 2:
                return np.polyfit(range(len(recent)), recent, 1)[0]  # slope
        elif metric == 'length' and self.episode_lengths:
            recent = list(self.episode_lengths)[-window:]
            if len(recent) >= 2:
                return np.polyfit(range(len(recent)), recent, 1)[0]  # slope
        
        return 0.0


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def create_logger(config):
    """Create logger based on configuration"""
    log_dir = config.get('log_dir', 'logs')
    use_tensorboard = config.get('use_tensorboard', False)
    use_wandb = config.get('use_wandb', False)
    
    return Logger(log_dir, use_tensorboard, use_wandb)


def setup_wandb(config):
    """Setup wandb logging"""
    try:
        import wandb
        
        wandb.init(
            project=config.get('project_name', 'drl-memory-maximize'),
            name=config.get('run_name', f"run_{int(time.time())}"),
            config=config,
            tags=config.get('tags', [])
        )
        
        return True
    except ImportError:
        print("wandb not available")
        return False