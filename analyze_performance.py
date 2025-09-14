import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse

# Import environment and agents
from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
from agents.a2c_diffusion import Diffusion_A2C
from agents.bc_diffusion import Diffusion_BC
from agents.gaussian_a2c import Gaussian_A2C
from agents.gaussian_dql import Gaussian_DQL
from agents.gaussian_ppo import Gaussian_PPO
from agents.ppo_diffusion import Diffusion_PPO
from agents.ql_diffusion import Diffusion_QL
from utils.replay_buffer import ReplayBuffer

def set_seed(seed):
    """Set random seeds for reproducibility"""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_agent(agent_name, state_dim, action_dim, max_action, device, **kwargs):
    """Create agent based on name"""
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
    
    agent_class = agents[agent_name]
    return agent_class(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        **kwargs
    )

def evaluate_agent_performance(agent_name, num_users, device, episodes=50, max_steps=1000):
    """Evaluate agent performance for a specific number of users"""
    print(f"Evaluating {agent_name} with {num_users} users...")
    
    # Create environment with modified number of users
    config = EnvConfig_v1("GAIServiceEnv")
    config["num_users"] = num_users
    
    env = GAIServiceEnv_v1(config, seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create agent with proper parameters for each agent type
    agent_params = {}
    
    if agent_name == 'gaussian_ppo':
        agent_params = {
            'discount': 0.99,
            'tau': 0.005,
            'lr': 7e-3,
            'grad_norm': 1.0,
            'clip_ratio': 0.2,
            'value_clip_ratio': 0.2,
            'norm_adv': True,
            'ent_coef': 0.01,
            'noise_scale': 0.1,
            'noise_type': 'gaussian',
            'epsilon': 0.1,
        }
    elif agent_name == 'gaussian_a2c':
        agent_params = {
            'lr': 3e-4,
            'gamma': 0.99,
            'value_coef': 0.5,
            'ent_coef': 0.01,
            'grad_norm': 1.0,
        }
    elif agent_name == 'gaussian_dql':
        agent_params = {
            'discount': 0.99,
            'tau': 0.005,
            'lr': 3e-4,
            'grad_norm': 1.0,
            'noise_scale': 0.3,
            'noise_type': 'gaussian',
            'epsilon': 0.01,
            'policy_delay': 2,
        }
    elif agent_name == 'a2c_diffusion':
        agent_params = {
            'gamma': 0.99,
            'lr': 1e-4,
            'n_timesteps': 20,
            'ema_decay': 0.995,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.25,
            'grad_norm': 0.25,
        }
    elif agent_name == 'bc_diffusion':
        agent_params = {
            'discount': 0.99,
            'tau': 0.005,
            'n_timesteps': 100,
            'lr': 2e-4,
        }
    elif agent_name == 'ql_diffusion':
        agent_params = {
            'discount': 0.99,
            'tau': 0.005,
            'n_timesteps': 100,
            'ema_decay': 0.995,
            'lr': 3e-4,
            'grad_norm': 1.0,
            'eta': 1.0,
        }
    
    agent = create_agent(agent_name, state_dim, action_dim, max_action, device, **agent_params)
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=10000, device=device)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    memory_usage = []
    latency_values = []
    qos_values = []
    denoise_steps = []
    
    # Training loop
    total_steps = 0
    episode = 0
    
    while total_steps < max_steps and episode < episodes:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Episode loop
        while not done and episode_length < 1000:
            # Sample action
            try:
                action = agent.sample_action(state, deterministic=True)
            except TypeError:
                action = agent.sample_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            replay_buffer.add(state, action, next_state, reward, done)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Train agent if buffer has enough samples
            if replay_buffer.size() >= 100 and total_steps % 10 == 0:
                try:
                    agent.train(replay_buffer, iterations=1, batch_size=32)
                except:
                    pass  # Skip training errors
        
        # Episode finished
        episode += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Extract metrics from info
        if 'per_user' in info:
            per_user = info['per_user']
            if per_user['mem']:
                memory_usage.append(np.mean(per_user['mem']))
            if per_user['latency']:
                latency_values.append(np.mean(per_user['latency']))
            if per_user['qos']:
                qos_values.append(np.mean(per_user['qos']))
            if per_user['steps']:
                denoise_steps.append(np.mean(per_user['steps']))
        
        if episode % 10 == 0:
            print(f"  Episode {episode}, Reward: {episode_reward:.2f}")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'memory': memory_usage,
        'latency': latency_values,
        'qos': qos_values,
        'denoise_steps': denoise_steps,
        'final_reward': np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        'avg_memory': np.mean(memory_usage) if memory_usage else 0,
        'avg_latency': np.mean(latency_values) if latency_values else 0,
        'avg_qos': np.mean(qos_values) if qos_values else 0,
        'avg_denoise_steps': np.mean(denoise_steps) if denoise_steps else 0,
    }

def plot_convergence_rewards(results, save_path='convergence_rewards.png'):
    """Plot reward convergence for all agents"""
    plt.figure(figsize=(12, 8))
    
    agent_names = list(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    for i, (agent_name, data) in enumerate(results.items()):
        if 'rewards' in data and data['rewards']:
            rewards = data['rewards']
            # Smooth the curve using moving average
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

def plot_users_analysis(results_by_users, save_dir='analysis_plots'):
    """Plot analysis for different number of users"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract data for plotting
    user_counts = sorted(results_by_users.keys())
    agent_names = list(next(iter(results_by_users.values())).keys())
    
    # 1. Reward vs Number of Users
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
    
    # 2. Memory Usage vs Number of Users
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
    
    # 3. Latency vs Number of Users
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
    
    # 4. QoS vs Number of Users
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
    
    # 5. Denoise Steps vs Number of Users
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
    
    # 6. Comprehensive comparison table
    create_comparison_table(results_by_users, f'{save_dir}/comprehensive_analysis.csv')
    
    print(f"All analysis plots saved to: {save_dir}/")

def create_comparison_table(results_by_users, save_path):
    """Create comprehensive comparison table"""
    data = []
    
    for users in sorted(results_by_users.keys()):
        for agent_name, metrics in results_by_users[users].items():
            data.append({
                'Users': users,
                'Agent': agent_name,
                'Final_Reward': metrics['final_reward'],
                'Avg_Memory': metrics['avg_memory'],
                'Avg_Latency': metrics['avg_latency'],
                'Avg_QoS': metrics['avg_qos'],
                'Avg_Denoise_Steps': metrics['avg_denoise_steps']
            })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Comparison table saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Analyze agent performance across different user counts')
    parser.add_argument('--agents', nargs='+', default=['gaussian_ppo', 'gaussian_a2c', 'gaussian_dql', 'a2c_diffusion', 'bc_diffusion', 'ql_diffusion'],
                       help='Agents to analyze')
    parser.add_argument('--user_counts', nargs='+', type=int, default=[8, 10, 12],
                       help='Number of users to test')
    parser.add_argument('--episodes', type=int, default=50,
                       help='Number of episodes per evaluation')
    parser.add_argument('--max_steps', type=int, default=1000,
                       help='Maximum training steps')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        import torch
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Results storage
    results_by_users = {}
    convergence_results = {}
    
    # Test each number of users
    for num_users in args.user_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {num_users} users")
        print(f"{'='*60}")
        
        results_by_users[num_users] = {}
        
        for agent_name in args.agents:
            try:
                metrics = evaluate_agent_performance(
                    agent_name, num_users, device, 
                    episodes=args.episodes, max_steps=args.max_steps
                )
                results_by_users[num_users][agent_name] = metrics
                
                # Store convergence data for the default user count (10)
                if num_users == 10:
                    convergence_results[agent_name] = metrics
                    
                print(f"{agent_name}: Final Reward = {metrics['final_reward']:.2f}")
                
            except Exception as e:
                print(f"Error evaluating {agent_name} with {num_users} users: {e}")
                continue
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating Analysis Plots")
    print(f"{'='*60}")
    
    # 1. Convergence plot
    if convergence_results:
        plot_convergence_rewards(convergence_results)
    
    # 2. User count analysis
    if results_by_users:
        plot_users_analysis(results_by_users)
    
    # Save raw results
    with open('performance_analysis_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_numpy(results_by_users), f, indent=2)
    
    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"{'='*60}")
    print("Generated plots:")
    print("- convergence_rewards.png: Reward convergence comparison")
    print("- analysis_plots/reward_vs_users.png: Reward vs number of users")
    print("- analysis_plots/memory_vs_users.png: Memory usage vs number of users")
    print("- analysis_plots/latency_vs_users.png: Latency vs number of users")
    print("- analysis_plots/qos_vs_users.png: QoS vs number of users")
    print("- analysis_plots/denoise_steps_vs_users.png: Denoise steps vs number of users")
    print("- analysis_plots/comprehensive_analysis.csv: Detailed comparison table")
    print("- performance_analysis_results.json: Raw results data")

if __name__ == "__main__":
    main()