import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Optional seaborn
try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False

# Env and agents
from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
from agents.a2c_diffusion import Diffusion_A2C
from agents.bc_diffusion import Diffusion_BC
from agents.gaussian_a2c import Gaussian_A2C
from agents.gaussian_dql import Gaussian_DQL
from agents.gaussian_ppo import Gaussian_PPO
from agents.ppo_diffusion import Diffusion_PPO
from agents.ql_diffusion import Diffusion_QL
from utils.replay_buffer import ReplayBuffer


# ---------------------- Common utilities ----------------------
def set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_agent(agent_name: str, state_dim: int, action_dim: int, max_action: float, device, **kwargs):
    agents = {
        'a2c_diffusion': Diffusion_A2C,
        'bc_diffusion': Diffusion_BC,
        'gaussian_a2c': Gaussian_A2C,
        'gaussian_dql': Gaussian_DQL,
        'gaussian_ppo': Gaussian_PPO,
        'ppo_diffusion': Diffusion_PPO,
        'ql_diffusion': Diffusion_QL,
    }
    if agent_name not in agents:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agents[agent_name](
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        **kwargs,
    )


def _agent_params(agent_name: str) -> Dict[str, Any]:
    if agent_name == 'gaussian_ppo':
        return {
            'discount': 0.99, 'tau': 0.005, 'lr': 7e-3, 'grad_norm': 1.0,
            'clip_ratio': 0.2, 'value_clip_ratio': 0.2, 'norm_adv': True,
            'ent_coef': 0.01, 'noise_scale': 0.1, 'noise_type': 'gaussian', 'epsilon': 0.1,
        }
    if agent_name == 'gaussian_a2c':
        return {'lr': 3e-4, 'gamma': 0.99, 'value_coef': 0.5, 'ent_coef': 0.01, 'grad_norm': 1.0}
    if agent_name == 'gaussian_dql':
        return {
            'discount': 0.99, 'tau': 0.005, 'lr': 3e-4, 'grad_norm': 1.0,
            'noise_scale': 0.3, 'noise_type': 'gaussian', 'epsilon': 0.01, 'policy_delay': 2,
        }
    if agent_name == 'a2c_diffusion':
        return {'gamma': 0.99, 'lr': 1e-4, 'n_timesteps': 20, 'ema_decay': 0.995, 'entropy_coef': 0.01, 'value_loss_coef': 0.25, 'grad_norm': 0.25}
    if agent_name == 'bc_diffusion':
        return {'discount': 0.99, 'tau': 0.005, 'n_timesteps': 100, 'lr': 2e-4}
    if agent_name == 'ql_diffusion':
        return {'discount': 0.99, 'tau': 0.005, 'n_timesteps': 100, 'ema_decay': 0.995, 'lr': 3e-4, 'grad_norm': 1.0, 'eta': 1.0}
    return {}


# ---------------------- Analyze performance (per users) ----------------------
def evaluate_agent_performance(agent_name: str, num_users: int, device, episodes: int = 50, max_steps: int = 1000):
    print(f"Evaluating {agent_name} with {num_users} users...")

    config = EnvConfig_v1("GAIServiceEnv")
    config["num_users"] = num_users
    env = GAIServiceEnv_v1(config, seed=42)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = create_agent(agent_name, state_dim, action_dim, max_action, device, **_agent_params(agent_name))
    replay_buffer = ReplayBuffer(max_size=10000, device=device)

    episode_rewards, episode_lengths = [], []
    memory_usage, latency_values, qos_values, denoise_steps = [], [], [], []
    total_flops_values = []

    total_steps = 0
    episode = 0
    while total_steps < max_steps and episode < episodes:
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        while not done and episode_length < 1000:
            try:
                action = agent.sample_action(state, deterministic=True)
            except TypeError:
                action = agent.sample_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if replay_buffer.size() >= 100 and total_steps % 10 == 0:
                try:
                    agent.train(replay_buffer, iterations=1, batch_size=32)
                except Exception:
                    pass

        episode += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if 'per_user' in info:
            per_user = info['per_user']
            serves = per_user.get('serve', [])
            mems = per_user.get('mem', [])
            lats = per_user.get('latency', [])
            qoss = per_user.get('qos', [])
            steps = per_user.get('steps', [])

            # Indices/users that were actually served
            served_indices = [i for i, s in enumerate(serves) if s > 0]

            if served_indices:
                # Memory: sum over served users
                memory_usage.append(float(np.sum([mems[i] for i in served_indices])))
                # Latency: system latency = max latency among served users
                latency_values.append(float(np.max([lats[i] for i in served_indices])))
                # QoS: average among served users
                qos_values.append(float(np.mean([qoss[i] for i in served_indices])))
                # Denoise steps: average among served users
                denoise_steps.append(float(np.mean([steps[i] for i in served_indices])))
            else:
                memory_usage.append(0.0)
                latency_values.append(0.0)
                qos_values.append(0.0)
                denoise_steps.append(0.0)

        # Track total FLOPs if provided by env
        total_flops_values.append(float(info.get('total_flops', 0)))

        if episode % 10 == 0:
            print(f"  Episode {episode}, Reward: {episode_reward:.2f}")

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'memory': memory_usage,
        'latency': latency_values,
        'qos': qos_values,
        'denoise_steps': denoise_steps,
        'final_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards) if episode_rewards else 0,
        'avg_memory': np.mean(memory_usage) if memory_usage else 0,
        'avg_latency': np.mean(latency_values) if latency_values else 0,
        'avg_qos': np.mean(qos_values) if qos_values else 0,
        'avg_denoise_steps': np.mean(denoise_steps) if denoise_steps else 0,
        'avg_total_flops': np.mean(total_flops_values) if total_flops_values else 0,
    }


def plot_convergence_rewards(results: Dict[str, Dict[str, Any]], save_path: str = 'convergence_rewards.png'):
    plt.figure(figsize=(12, 8))
    agent_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    for i, (agent_name, data) in enumerate(results.items()):
        if 'rewards' in data and data['rewards']:
            rewards = data['rewards']
            window_size = max(1, len(rewards) // 20)
            if window_size > 1:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                episodes = np.arange(len(smoothed_rewards))
            else:
                smoothed_rewards = rewards
                episodes = np.arange(len(rewards))
            plt.plot(episodes, smoothed_rewards, label=agent_name, color=colors[i], linewidth=2, alpha=0.8)
    plt.title('Reward Convergence Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Convergence plot saved to: {save_path}")


def create_comparison_table(results_by_users: Dict[int, Dict[str, Any]], save_path: str):
    data_rows = []
    for users in sorted(results_by_users.keys()):
        for agent_name, metrics in results_by_users[users].items():
            data_rows.append({
                'Users': users,
                'Agent': agent_name,
                'Final_Reward': metrics.get('final_reward', 0),
                'Avg_Memory': metrics.get('avg_memory', 0),
                'Avg_Latency': metrics.get('avg_latency', 0),
                'Avg_QoS': metrics.get('avg_qos', 0),
                'Avg_Denoise_Steps': metrics.get('avg_denoise_steps', 0),
            })
    df = pd.DataFrame(data_rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Comparison table saved to: {save_path}")


def plot_users_analysis(results_by_users: Dict[int, Dict[str, Any]], save_dir: str = 'analysis_plots'):
    os.makedirs(save_dir, exist_ok=True)
    user_counts = sorted(results_by_users.keys())
    if not user_counts:
        print("No results to plot for users analysis.")
        return
    agent_names = list(next(iter(results_by_users.values())).keys())

    # Reward vs Users
    plt.figure(figsize=(10, 6))
    for agent_name in agent_names:
        rewards = [results_by_users[users][agent_name]['final_reward'] for users in user_counts]
        plt.plot(user_counts, rewards, marker='o', label=agent_name, linewidth=2, markersize=8)
    plt.title('Final Reward vs Number of Users', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Users')
    plt.ylabel('Final Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/reward_vs_users.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Memory vs Users
    plt.figure(figsize=(10, 6))
    for agent_name in agent_names:
        memory = [results_by_users[users][agent_name]['avg_memory'] for users in user_counts]
        plt.plot(user_counts, memory, marker='s', label=agent_name, linewidth=2, markersize=8)
    plt.title('Average Memory Usage vs Number of Users', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Users')
    plt.ylabel('Average Memory Usage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/memory_vs_users.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Latency vs Users
    plt.figure(figsize=(10, 6))
    for agent_name in agent_names:
        latency = [results_by_users[users][agent_name]['avg_latency'] for users in user_counts]
        plt.plot(user_counts, latency, marker='^', label=agent_name, linewidth=2, markersize=8)
    plt.title('Average Latency vs Number of Users', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Users')
    plt.ylabel('Average Latency (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/latency_vs_users.png', dpi=300, bbox_inches='tight')
    plt.show()

    # QoS vs Users
    plt.figure(figsize=(10, 6))
    for agent_name in agent_names:
        qos = [results_by_users[users][agent_name]['avg_qos'] for users in user_counts]
        plt.plot(user_counts, qos, marker='d', label=agent_name, linewidth=2, markersize=8)
    plt.title('Average QoS vs Number of Users', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Users')
    plt.ylabel('Average QoS (BRISQUE score)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/qos_vs_users.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Denoise Steps vs Users
    plt.figure(figsize=(10, 6))
    for agent_name in agent_names:
        steps = [results_by_users[users][agent_name]['avg_denoise_steps'] for users in user_counts]
        plt.plot(user_counts, steps, marker='v', label=agent_name, linewidth=2, markersize=8)
    plt.title('Average Denoise Steps vs Number of Users', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Users')
    plt.ylabel('Average Denoise Steps')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/denoise_steps_vs_users.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save table
    create_comparison_table(results_by_users, f'{save_dir}/comprehensive_analysis.csv')
    print(f"All analysis plots saved to: {save_dir}/")


# ---------------------- Convergence collection/plots ----------------------
def collect_convergence_data(agent_name: str, num_users: int, device, episodes: int = 100, max_steps: int = 2000):
    print(f"Collecting convergence data for {agent_name} with {num_users} users...")
    config = EnvConfig_v1("GAIServiceEnv")
    config["num_users"] = num_users
    env = GAIServiceEnv_v1(config, seed=42)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = create_agent(agent_name, state_dim, action_dim, max_action, device, **_agent_params(agent_name))
    replay_buffer = ReplayBuffer(max_size=10000, device=device)

    data = {
        'episodes': [], 'rewards': [], 'memory': [], 'latency': [], 'qos': [],
        'denoise_steps': [], 'served_users': [], 'total_latency': [], 'total_memory': [],
        'total_flops': [], 'penalty': [], 'bonus': []
    }

    total_steps = 0
    episode = 0
    while total_steps < max_steps and episode < episodes:
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        while not done and episode_length < 1000:
            try:
                action = agent.sample_action(state, deterministic=True)
            except TypeError:
                action = agent.sample_action(state)

            next_state, reward, done, info = env.step(action)
            replay_buffer.add(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if replay_buffer.size() >= 100 and total_steps % 10 == 0:
                try:
                    agent.train(replay_buffer, iterations=1, batch_size=32)
                except Exception:
                    pass

        episode += 1

        if 'per_user' in info:
            per_user = info['per_user']
            serves = per_user.get('serve', [])
            mems = per_user.get('mem', [])
            lats = per_user.get('latency', [])
            qoss = per_user.get('qos', [])
            steps = per_user.get('steps', [])
            served_indices = [i for i, s in enumerate(serves) if s > 0]
            served_count = len(served_indices)
            if served_indices:
                # Sum memory across served users
                avg_memory = float(np.sum([mems[i] for i in served_indices]))
                # System latency is max among served users
                avg_latency = float(np.max([lats[i] for i in served_indices]))
                # Average QoS and denoise steps among served users
                avg_qos = float(np.mean([qoss[i] for i in served_indices]))
                avg_steps = float(np.mean([steps[i] for i in served_indices]))
            else:
                avg_memory = avg_latency = avg_qos = avg_steps = 0.0
        else:
            served_count = 0
            avg_memory = avg_latency = avg_qos = avg_steps = 0.0

        data['episodes'].append(episode)
        data['rewards'].append(episode_reward)
        data['memory'].append(avg_memory)
        data['latency'].append(avg_latency)
        data['qos'].append(avg_qos)
        data['denoise_steps'].append(avg_steps)
        data['served_users'].append(served_count)
        data['total_latency'].append(info.get('total_latency', 0))
        data['total_memory'].append(info.get('total_mem', 0))
        data['total_flops'].append(info.get('total_flops', 0))
        data['penalty'].append(info.get('penalty', 0))
        data['bonus'].append(info.get('bonus', 0))

        if episode % 10 == 0:
            print(f"  Episode {episode}, Reward: {episode_reward:.2f}, Served: {served_count}")

    return data


def plot_convergence_metrics(convergence_data: Dict[str, Any], agent_name: str, save_dir: str = 'convergence_plots'):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'{agent_name.replace("_", " ").title()} - Convergence Analysis', fontsize=16, fontweight='bold')

    episodes = convergence_data['episodes']
    window_size = max(1, len(episodes) // 20)

    # 1 Reward
    ax1 = axes[0, 0]
    ax1.plot(episodes, convergence_data['rewards'], 'b-', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['rewards'], np.ones(window_size)/window_size, mode='valid')
        ax1.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})')
        ax1.legend()
    ax1.set_title('Reward Convergence'); ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward'); ax1.grid(True, alpha=0.3)

    # 2 Memory
    ax2 = axes[0, 1]
    ax2.plot(episodes, convergence_data['memory'], 'g-', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['memory'], np.ones(window_size)/window_size, mode='valid')
        ax2.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})'); ax2.legend()
    ax2.set_title('Memory Usage Convergence'); ax2.set_xlabel('Episode'); ax2.set_ylabel('Average Memory'); ax2.grid(True, alpha=0.3)

    # 3 Latency
    ax3 = axes[0, 2]
    ax3.plot(episodes, convergence_data['latency'], 'orange', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['latency'], np.ones(window_size)/window_size, mode='valid')
        ax3.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})'); ax3.legend()
    ax3.set_title('Latency Convergence'); ax3.set_xlabel('Episode'); ax3.set_ylabel('Average Latency (s)'); ax3.grid(True, alpha=0.3)

    # 4 QoS
    ax4 = axes[1, 0]
    ax4.plot(episodes, convergence_data['qos'], 'purple', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['qos'], np.ones(window_size)/window_size, mode='valid')
        ax4.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})'); ax4.legend()
    ax4.set_title('QoS Convergence'); ax4.set_xlabel('Episode'); ax4.set_ylabel('Average QoS (BRISQUE)'); ax4.grid(True, alpha=0.3)

    # 5 Denoise steps
    ax5 = axes[1, 1]
    ax5.plot(episodes, convergence_data['denoise_steps'], 'brown', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['denoise_steps'], np.ones(window_size)/window_size, mode='valid')
        ax5.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})'); ax5.legend()
    ax5.set_title('Denoise Steps Convergence'); ax5.set_xlabel('Episode'); ax5.set_ylabel('Average Denoise Steps'); ax5.grid(True, alpha=0.3)

    # 6 Served users
    ax6 = axes[1, 2]
    ax6.plot(episodes, convergence_data['served_users'], 'pink', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['served_users'], np.ones(window_size)/window_size, mode='valid')
        ax6.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})'); ax6.legend()
    ax6.set_title('Served Users Convergence'); ax6.set_xlabel('Episode'); ax6.set_ylabel('Number of Served Users'); ax6.grid(True, alpha=0.3)

    # 7 Total latency
    ax7 = axes[2, 0]
    ax7.plot(episodes, convergence_data['total_latency'], 'cyan', linewidth=2, alpha=0.7)
    if window_size > 1:
        ma = np.convolve(convergence_data['total_latency'], np.ones(window_size)/window_size, mode='valid')
        ax7.plot(episodes[window_size-1:], ma, 'r-', linewidth=3, label=f'MA({window_size})'); ax7.legend()
    ax7.set_title('Total System Latency'); ax7.set_xlabel('Episode'); ax7.set_ylabel('Total Latency (s)'); ax7.grid(True, alpha=0.3)

    # 8 Penalty/Bonus
    ax8 = axes[2, 1]
    ax8.plot(episodes, convergence_data['penalty'], 'red', linewidth=2, alpha=0.7, label='Penalty')
    ax8.plot(episodes, convergence_data['bonus'], 'green', linewidth=2, alpha=0.7, label='Bonus')
    ax8.set_title('Penalty and Bonus'); ax8.set_xlabel('Episode'); ax8.set_ylabel('Value'); ax8.legend(); ax8.grid(True, alpha=0.3)

    # 9 Reward distribution
    ax9 = axes[2, 2]
    ax9.hist(convergence_data['rewards'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    if convergence_data['rewards']:
        ax9.axvline(np.mean(convergence_data['rewards']), color='red', linestyle='--', linewidth=2, label=f"Mean: {np.mean(convergence_data['rewards']):.0f}")
        ax9.axvline(np.median(convergence_data['rewards']), color='orange', linestyle='--', linewidth=2, label=f"Median: {np.median(convergence_data['rewards']):.0f}")
        ax9.legend()
    ax9.set_title('Reward Distribution'); ax9.set_xlabel('Reward'); ax9.set_ylabel('Frequency'); ax9.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{agent_name}_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Convergence plot for {agent_name} saved to: {save_dir}/{agent_name}_convergence.png")


def plot_comparative_convergence(all_convergence_data: Dict[str, Dict[str, Any]], save_dir: str = 'convergence_plots'):
    os.makedirs(save_dir, exist_ok=True)
    agent_colors = {
        'gaussian_ppo': '#1f77b4', 'gaussian_a2c': '#ff7f0e', 'gaussian_dql': '#2ca02c',
        'a2c_diffusion': '#d62728', 'bc_diffusion': '#9467bd', 'ql_diffusion': '#8c564b',
    }
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparative Convergence Analysis - All Agents', fontsize=16, fontweight='bold')

    # Reward
    ax1 = axes[0, 0]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        rewards = data['rewards']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(episodes[window_size-1:], smoothed, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax1.plot(episodes, rewards, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
    ax1.set_title('Reward Convergence Comparison'); ax1.set_xlabel('Episode'); ax1.set_ylabel('Reward'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Memory
    ax2 = axes[0, 1]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        memory = data['memory']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(memory, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(episodes[window_size-1:], smoothed, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax2.plot(episodes, memory, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
    ax2.set_title('Memory Usage Convergence'); ax2.set_xlabel('Episode'); ax2.set_ylabel('Average Memory'); ax2.legend(); ax2.grid(True, alpha=0.3)

    # Latency
    ax3 = axes[0, 2]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        latency = data['latency']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(latency, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(episodes[window_size-1:], smoothed, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax3.plot(episodes, latency, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
    ax3.set_title('Latency Convergence'); ax3.set_xlabel('Episode'); ax3.set_ylabel('Average Latency (s)'); ax3.legend(); ax3.grid(True, alpha=0.3)

    # QoS
    ax4 = axes[1, 0]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        qos = data['qos']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(qos, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(episodes[window_size-1:], smoothed, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax4.plot(episodes, qos, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
    ax4.set_title('QoS Convergence'); ax4.set_xlabel('Episode'); ax4.set_ylabel('Average QoS (BRISQUE)'); ax4.legend(); ax4.grid(True, alpha=0.3)

    # Denoise steps
    ax5 = axes[1, 1]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        steps = data['denoise_steps']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
            ax5.plot(episodes[window_size-1:], smoothed, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax5.plot(episodes, steps, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
    ax5.set_title('Denoise Steps Convergence'); ax5.set_xlabel('Episode'); ax5.set_ylabel('Average Denoise Steps'); ax5.legend(); ax5.grid(True, alpha=0.3)

    # Served users
    ax6 = axes[1, 2]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        served = data['served_users']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(served, np.ones(window_size)/window_size, mode='valid')
            ax6.plot(episodes[window_size-1:], smoothed, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax6.plot(episodes, served, color=agent_colors.get(agent_name, 'black'), linewidth=2, label=agent_name.replace('_', ' ').title())
    ax6.set_title('Served Users Convergence'); ax6.set_xlabel('Episode'); ax6.set_ylabel('Number of Served Users'); ax6.legend(); ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparative_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Comparative convergence plot saved to: {save_dir}/comparative_convergence.png")


# ---------------------- Comprehensive dashboard (from CSV) ----------------------
def create_comprehensive_analysis(csv_path: str = 'analysis_plots/comprehensive_analysis.csv'):
    df = pd.read_csv(csv_path)
    plt.style.use('default')

    fig = plt.figure(figsize=(20, 15))
    agent_colors = {
        'gaussian_ppo': '#1f77b4', 'gaussian_a2c': '#ff7f0e', 'gaussian_dql': '#2ca02c',
        'a2c_diffusion': '#d62728', 'bc_diffusion': '#9467bd', 'ql_diffusion': '#8c564b'
    }

    # 1 Reward vs Users
    ax1 = plt.subplot(3, 3, 1)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax1.plot(agent_data['Users'], agent_data['Final_Reward'], marker='o', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax1.set_title('Reward vs Number of Users', fontsize=14, fontweight='bold'); ax1.set_xlabel('Number of Users'); ax1.set_ylabel('Final Reward'); ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); ax1.grid(True, alpha=0.3)

    # 2 Memory vs Users
    ax2 = plt.subplot(3, 3, 2)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax2.plot(agent_data['Users'], agent_data['Avg_Memory'], marker='s', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax2.set_title('Memory Usage vs Number of Users', fontsize=14, fontweight='bold'); ax2.set_xlabel('Number of Users'); ax2.set_ylabel('Average Memory Usage'); ax2.grid(True, alpha=0.3)

    # 3 Latency vs Users
    ax3 = plt.subplot(3, 3, 3)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax3.plot(agent_data['Users'], agent_data['Avg_Latency'], marker='^', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax3.set_title('Latency vs Number of Users', fontsize=14, fontweight='bold'); ax3.set_xlabel('Number of Users'); ax3.set_ylabel('Average Latency (seconds)'); ax3.grid(True, alpha=0.3)

    # 4 QoS vs Users
    ax4 = plt.subplot(3, 3, 4)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax4.plot(agent_data['Users'], agent_data['Avg_QoS'], marker='d', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax4.set_title('QoS vs Number of Users', fontsize=14, fontweight='bold'); ax4.set_xlabel('Number of Users'); ax4.set_ylabel('Average QoS (BRISQUE score)'); ax4.grid(True, alpha=0.3)

    # 5 Denoise steps
    ax5 = plt.subplot(3, 3, 5)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax5.plot(agent_data['Users'], agent_data['Avg_Denoise_Steps'], marker='v', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax5.set_title('Denoise Steps vs Number of Users', fontsize=14, fontweight='bold'); ax5.set_xlabel('Number of Users'); ax5.set_ylabel('Average Denoise Steps'); ax5.grid(True, alpha=0.3)

    # 6 Performance heatmap
    ax6 = plt.subplot(3, 3, 6)
    pivot_table = df.pivot(index='Agent', columns='Users', values='Final_Reward')
    if HAS_SEABORN:
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax6, cbar_kws={'label': 'Final Reward'})
    else:
        im = ax6.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
        ax6.set_xticks(range(len(pivot_table.columns))); ax6.set_yticks(range(len(pivot_table.index)))
        ax6.set_xticklabels(pivot_table.columns); ax6.set_yticklabels(pivot_table.index)
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                ax6.text(j, i, f'{pivot_table.iloc[i, j]:.0f}', ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax6, label='Final Reward')
    ax6.set_title('Performance Heatmap', fontsize=14, fontweight='bold'); ax6.set_xlabel('Number of Users'); ax6.set_ylabel('Agent')

    # 7 Memory efficiency
    ax7 = plt.subplot(3, 3, 7)
    df['Memory_Efficiency'] = df['Final_Reward'] / df['Avg_Memory'].replace(0, np.nan)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax7.plot(agent_data['Users'], agent_data['Memory_Efficiency'], marker='o', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax7.set_title('Memory Efficiency vs Number of Users', fontsize=14, fontweight='bold'); ax7.set_xlabel('Number of Users'); ax7.set_ylabel('Reward per Memory Unit'); ax7.grid(True, alpha=0.3)

    # 8 Latency efficiency
    ax8 = plt.subplot(3, 3, 8)
    df['Latency_Efficiency'] = df['Final_Reward'] / (df['Avg_Latency'] + 1e-6)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax8.plot(agent_data['Users'], agent_data['Latency_Efficiency'], marker='s', linewidth=2, markersize=8, label=agent.replace('_', ' ').title(), color=agent_colors.get(agent, 'black'))
    ax8.set_title('Latency Efficiency vs Number of Users', fontsize=14, fontweight='bold'); ax8.set_xlabel('Number of Users'); ax8.set_ylabel('Reward per Latency Unit'); ax8.grid(True, alpha=0.3)

    # 9 Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = "PERFORMANCE SUMMARY\n\n"
    best_overall = df.loc[df['Final_Reward'].idxmax()]
    summary_text += f"Best Overall: {best_overall['Agent'].replace('_', ' ').title()}\n"
    summary_text += f"Reward: {best_overall['Final_Reward']:.0f}\n"
    summary_text += f"Users: {int(best_overall['Users'])}\n\n"
    best_memory = df.loc[df['Memory_Efficiency'].idxmax()]
    summary_text += f"Most Memory Efficient:\n{best_memory['Agent'].replace('_', ' ').title()}\n"
    summary_text += f"Efficiency: {best_memory['Memory_Efficiency']:.0f}\n\n"
    best_latency = df.loc[df['Latency_Efficiency'].idxmax()]
    summary_text += f"Most Latency Efficient:\n{best_latency['Agent'].replace('_', ' ').title()}\n"
    summary_text += f"Efficiency: {best_latency['Latency_Efficiency']:.0f}\n\n"
    summary_text += "Avg Performance by Users:\n"
    for users in sorted(df['Users'].unique()):
        avg_reward = df[df['Users'] == users]['Final_Reward'].mean()
        summary_text += f"{int(users)} users: {avg_reward:.0f}\n"
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Comprehensive analysis plot saved as: comprehensive_analysis.png")


# ---------------------- CLI orchestration ----------------------
def main():
    parser = argparse.ArgumentParser(description='Unified convergence and performance analysis')
    parser.add_argument('--agents', nargs='+', default=['gaussian_ppo', 'gaussian_a2c', 'gaussian_dql', 'a2c_diffusion', 'bc_diffusion', 'ql_diffusion'], help='Agents to analyze')
    parser.add_argument('--num_users', type=int, default=10, help='Number of users for convergence analysis')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per run')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum training steps')
    parser.add_argument('--user_counts', nargs='+', type=int, default=[8, 10, 12], help='User counts for performance analysis')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default='convergence_plots', help='Directory to save plots')
    parser.add_argument('--run_performance', action='store_true', help='Run performance analysis across user counts')
    parser.add_argument('--run_convergence', action='store_true', help='Collect per-agent convergence data and plots')
    parser.add_argument('--run_dashboard', action='store_true', help='Build comprehensive analysis dashboard from CSV')
    parser.add_argument('--replot', action='store_true', help='Rebuild plots from saved JSONs; skip running env/agents')

    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        import torch
        device = torch.device(args.device)
    print(f"Using device: {device}")

    set_seed(args.seed)

    # Save run configuration for reproducibility and re-plotting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_config = {
        'timestamp': timestamp,
        'args': vars(args),
    }
    try:
        with open('last_run_args.json', 'w') as f:
            json.dump(run_config, f, indent=2)
    except Exception:
        pass
    try:
        os.makedirs(args.save_dir, exist_ok=True)
        with open(os.path.join(args.save_dir, 'last_run_args.json'), 'w') as f:
            json.dump(run_config, f, indent=2)
    except Exception:
        pass
    try:
        os.makedirs('analysis_plots', exist_ok=True)
        with open(os.path.join('analysis_plots', 'last_run_args.json'), 'w') as f:
            json.dump(run_config, f, indent=2)
    except Exception:
        pass

    # Re-plot mode: regenerate figures from saved JSONs without running env/agents
    if args.replot:
        # Performance analysis re-plot
        if args.run_performance:
            # Prefer metrics_analyzed.json if available (already post-aggregated)
            src_file = 'metrics_analyzed.json' if os.path.isfile('metrics_analyzed.json') else 'performance_analysis_results.json'
            if os.path.isfile(src_file):
                try:
                    with open(src_file, 'r') as f:
                        results_by_users = json.load(f)
                    # Plot overall analysis figures
                    if results_by_users:
                        plot_users_analysis(results_by_users, save_dir='analysis_plots')
                        # If the selected num_users exists, build convergence rewards comparison from it
                        # Support both dict keys: int or str (depending on how it was saved)
                        if isinstance(results_by_users, dict):
                            if args.num_users in results_by_users:
                                plot_convergence_rewards(results_by_users[args.num_users], save_path='convergence_rewards.png')
                            elif str(args.num_users) in results_by_users:
                                plot_convergence_rewards(results_by_users[str(args.num_users)], save_path='convergence_rewards.png')
                except Exception as e:
                    print(f"Failed to replot performance analysis: {e}")
            else:
                print("No analyzed metrics file found; cannot replot performance analysis.")

        # Convergence analysis re-plot
        if args.run_convergence:
            convergence_json_path = os.path.join(args.save_dir, 'convergence_data.json')
            if os.path.isfile(convergence_json_path):
                try:
                    with open(convergence_json_path, 'r') as f:
                        all_convergence_data = json.load(f)
                    if isinstance(all_convergence_data, dict) and all_convergence_data:
                        # Per-agent plots
                        for agent_name, data in all_convergence_data.items():
                            plot_convergence_metrics(data, agent_name, args.save_dir)
                        # Comparative plot
                        plot_comparative_convergence(all_convergence_data, args.save_dir)
                except Exception as e:
                    print(f"Failed to replot convergence analysis: {e}")
            else:
                print(f"{convergence_json_path} not found; cannot replot convergence analysis.")

        return

    # Run performance across user counts
    results_by_users: Dict[int, Dict[str, Any]] = {}
    convergence_results: Dict[str, Dict[str, Any]] = {}

    if args.run_performance:
        for num_users in tqdm(args.user_counts):
            print(f"\n{'='*60}\nTesting with {num_users} users\n{'='*60}")
            results_by_users[num_users] = {}
            for i, agent_name in enumerate(args.agents):
                try:
                    metrics = evaluate_agent_performance(agent_name, num_users, device, episodes=args.episodes, max_steps=args.max_steps)
                    results_by_users[num_users][agent_name] = metrics
                    if num_users == args.num_users:
                        convergence_results[agent_name] = metrics
                    print(f"{agent_name}: Final Reward = {metrics['final_reward']:.2f} {i+1}/{len(args.agents)}")
                except Exception as e:
                    print(f"Error evaluating {agent_name} with {num_users} users: {e}")

        # Plots and CSV
        if convergence_results:
            plot_convergence_rewards(convergence_results, save_path='convergence_rewards.png')
        if results_by_users:
            plot_users_analysis(results_by_users, save_dir='analysis_plots')
        # Save analyzed metrics snapshot for later re-plotting
        try:
            with open('metrics_analyzed.json', 'w') as f:
                json.dump(results_by_users, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to save metrics_analyzed.json: {e}")
        with open('performance_analysis_results.json', 'w') as f:
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert_numpy(x) for x in obj]
                return obj
            json.dump(convert_numpy(results_by_users), f, indent=2)

    # Collect and plot convergence data per agent
    if args.run_convergence:
        all_convergence_data: Dict[str, Dict[str, Any]] = {}
        print(f"\n{'='*60}\nCOLLECTING CONVERGENCE DATA\n{'='*60}")
        for agent_name in tqdm(args.agents):
            try:
                data = collect_convergence_data(agent_name, args.num_users, device, episodes=args.episodes, max_steps=args.max_steps)
                all_convergence_data[agent_name] = data
                plot_convergence_metrics(data, agent_name, args.save_dir)
            except Exception as e:
                print(f"Error collecting data for {agent_name}: {e}")
        if all_convergence_data:
            plot_comparative_convergence(all_convergence_data, args.save_dir)
            with open(f'{args.save_dir}/convergence_data.json', 'w') as f:
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, dict):
                        return {k: convert_numpy(v) for k, v in obj.items()}
                    if isinstance(obj, list):
                        return [convert_numpy(x) for x in obj]
                    return obj
                json.dump(convert_numpy(all_convergence_data), f, indent=2)

    # Build comprehensive dashboard from CSV (generated by performance analysis)
    if args.run_dashboard:
        create_comprehensive_analysis('analysis_plots/comprehensive_analysis.csv')


if __name__ == "__main__":
    main()