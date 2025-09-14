# utils/metrics_logger.py
import os, json, csv
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MetricsLogger:
    def __init__(self, save_dir="metrics_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.epoch_metrics = []
        self.step_metrics = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.loss_history = {'epochs': [], 'losses': {}, 'rewards': []}
        self.best_reward = float('-inf')

    def log_epoch_metrics(self, epoch, rewards, agent_losses=None):
        if isinstance(rewards, (int, float)): rewards = [rewards]
        epoch_data = {
            'epoch': epoch,
            'avg_reward': float(np.mean(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'std_reward': float(np.std(rewards)),
            'total_episodes': int(len(rewards)),
            'timestamp': datetime.now().isoformat()
        }
        if agent_losses: epoch_data.update(agent_losses)
        self.epoch_metrics.append(epoch_data)
        self.loss_history['epochs'].append(epoch)
        avg_reward = epoch_data['avg_reward']
        self.loss_history['rewards'].append(avg_reward)
        reward_improved = False
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            reward_improved = True
        if agent_losses:
            for k, v in agent_losses.items():
                self.loss_history['losses'].setdefault(k, []).append(float(v))
        return reward_improved

    def _ensure_axes(self, num_plots):
        if num_plots == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 4))
            return fig, [axes]
        return plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

    def save_loss_plot(self):
        if not self.loss_history['epochs']: return
        num_plots = 1 + len(self.loss_history['losses'])
        fig, axes = self._ensure_axes(num_plots)
        # rewards
        axes[0].plot(self.loss_history['epochs'], self.loss_history['rewards'], linewidth=2, label='Average Reward')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Reward')
        axes[0].set_title('Training Progress - Average Reward'); axes[0].grid(True, alpha=0.3); axes[0].legend()
        # losses
        plot_idx = 1
        for loss_name, vals in self.loss_history['losses'].items():
            if not vals: continue
            ep = self.loss_history['epochs'][-len(vals):]
            axes[plot_idx].plot(ep, vals, linewidth=2, label=loss_name)
            axes[plot_idx].set_xlabel('Epoch'); axes[plot_idx].set_ylabel('Loss')
            axes[plot_idx].set_title(f'Training Progress - {loss_name}')
            axes[plot_idx].grid(True, alpha=0.3); axes[plot_idx].legend()
            plot_idx += 1
        for i in range(plot_idx, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout()
        path = os.path.join(self.save_dir, "loss_plot.png")
        plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"📊 Loss plot saved: {path}")

    def save_final_plot(self):
        if not self.loss_history['epochs']: return None
        num_plots = 1 + len(self.loss_history['losses'])
        fig, axes = self._ensure_axes(num_plots)
        epochs = self.loss_history['epochs']; rewards = self.loss_history['rewards']
        axes[0].plot(epochs, rewards, linewidth=1, alpha=0.7, label='Average Reward')
        if len(rewards) > 10:
            w = min(50, max(3, len(rewards)//10))
            ma = np.convolve(rewards, np.ones(w)/w, mode='valid')
            axes[0].plot(epochs[w-1:], ma, linewidth=2, label=f'MA({w})')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Reward')
        axes[0].set_title('Final Training Results - Reward'); axes[0].grid(True, alpha=0.3); axes[0].legend()
        if rewards:
            stats_text = f'Final: {rewards[-1]:.4f}\nBest: {max(rewards):.4f}\nAvg: {np.mean(rewards):.4f}\nStd: {np.std(rewards):.4f}'
            axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        idx = 1
        for name, vals in self.loss_history['losses'].items():
            if not vals: continue
            ep = epochs[-len(vals):]
            axes[idx].plot(ep, vals, linewidth=2, label=name)
            if len(vals) > 10:
                w = min(20, max(3, len(vals)//5))
                ma = np.convolve(vals, np.ones(w)/w, mode='valid')
                axes[idx].plot(ep[w-1:], ma, linewidth=2, label=f'MA({w})')
            axes[idx].set_xlabel('Epoch'); axes[idx].set_ylabel('Loss')
            axes[idx].set_title(f'Final Training Results - {name}')
            axes[idx].grid(True, alpha=0.3); axes[idx].legend()
            idx += 1
        for i in range(idx, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout()
        path = os.path.join(self.save_dir, f"final_training_plot_{self.timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight'); plt.close()
        print(f"🎯 Final training plot saved: {path}")
        return path

    def _convert(self, data):
        import numpy as np
        if isinstance(data, dict): return {k: self._convert(v) for k, v in data.items()}
        if isinstance(data, list): return [self._convert(x) for x in data]
        if isinstance(data, np.ndarray): return data.tolist()
        if isinstance(data, (np.float32, np.float64)): return float(data)
        if isinstance(data, (np.int32, np.int64)): return int(data)
        if isinstance(data, np.bool_): return bool(data)
        return data

    def save_to_csv(self):
        if self.epoch_metrics:
            path = os.path.join(self.save_dir, "epoch_metrics.csv")
            with open(path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=self.epoch_metrics[0].keys())
                w.writeheader(); w.writerows(self.epoch_metrics)
            print(f"Epoch metrics saved to: {path}")

    def save_to_json(self):
        if self.epoch_metrics:
            path = os.path.join(self.save_dir, "epoch_metrics.json")
            with open(path, 'w') as f:
                json.dump(self._convert(self.epoch_metrics), f, indent=2)
            print(f"Epoch metrics (JSON) saved: {path}")
        if self.step_metrics:
            path = os.path.join(self.save_dir, "step_metrics.json")
            with open(path, 'w') as f:
                json.dump(self._convert(self.step_metrics), f, indent=2)
            print(f"Step metrics (JSON) saved: {path}")
