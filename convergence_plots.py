import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import argparse
import glob

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

def collect_convergence_data(agent_name, num_users, device, episodes=100, max_steps=2000):
    """Collect detailed convergence data for an agent"""
    print(f"Collecting convergence data for {agent_name} with {num_users} users...")
    
    # Create environment
    config = EnvConfig_v1("GAIServiceEnv")
    config["num_users"] = num_users
    
    env = GAIServiceEnv_v1(config, seed=42)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create agent with proper parameters
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
    
    # Detailed metrics collection
    convergence_data = {
        'episodes': [],
        'rewards': [],
        'memory': [],
        'latency': [],
        'qos': [],
        'denoise_steps': [],
        'served_users': [],
        'total_latency': [],
        'total_memory': [],
        'total_flops': [],
        'penalty': [],
        'bonus': []
    }
    
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
        
        # Episode finished - collect metrics
        episode += 1
        
        # Extract metrics from info
        if 'per_user' in info:
            per_user = info['per_user']
            
            # Calculate averages for served users only
            served_mask = [s > 0 for s in per_user['serve']]
            if any(served_mask):
                avg_memory = np.mean([m for m, s in zip(per_user['mem'], per_user['serve']) if s > 0])
                avg_latency = np.mean([l for l, s in zip(per_user['latency'], per_user['serve']) if s > 0])
                avg_qos = np.mean([q for q, s in zip(per_user['qos'], per_user['serve']) if s > 0])
                avg_denoise_steps = np.mean([d for d, s in zip(per_user['steps'], per_user['serve']) if s > 0])
                served_count = sum(per_user['serve'])
            else:
                avg_memory = 0
                avg_latency = 0
                avg_qos = 0
                avg_denoise_steps = 0
                served_count = 0
        else:
            avg_memory = 0
            avg_latency = 0
            avg_qos = 0
            avg_denoise_steps = 0
            served_count = 0
        
        # Store convergence data
        convergence_data['episodes'].append(episode)
        convergence_data['rewards'].append(episode_reward)
        convergence_data['memory'].append(avg_memory)
        convergence_data['latency'].append(avg_latency)
        convergence_data['qos'].append(avg_qos)
        convergence_data['denoise_steps'].append(avg_denoise_steps)
        convergence_data['served_users'].append(served_count)
        
        # Store system-level metrics
        convergence_data['total_latency'].append(info.get('total_latency', 0))
        convergence_data['total_memory'].append(info.get('total_mem', 0))
        convergence_data['total_flops'].append(info.get('total_flops', 0))
        convergence_data['penalty'].append(info.get('penalty', 0))
        convergence_data['bonus'].append(info.get('bonus', 0))
        
        if episode % 10 == 0:
            print(f"  Episode {episode}, Reward: {episode_reward:.2f}, Served: {served_count}")
    
    return convergence_data

def plot_convergence_metrics(convergence_data, agent_name, save_dir='convergence_plots'):
    """Plot detailed convergence metrics for a single agent"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(f'{agent_name.replace("_", " ").title()} - Convergence Analysis', 
                 fontsize=16, fontweight='bold')
    
    episodes = convergence_data['episodes']
    
    # 1. Reward convergence
    ax1 = axes[0, 0]
    ax1.plot(episodes, convergence_data['rewards'], 'b-', linewidth=2, alpha=0.7)
    # Add moving average
    window_size = max(1, len(episodes) // 20)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['rewards'], np.ones(window_size)/window_size, mode='valid')
        ax1.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax1.legend()
    ax1.set_title('Reward Convergence')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory usage convergence
    ax2 = axes[0, 1]
    ax2.plot(episodes, convergence_data['memory'], 'g-', linewidth=2, alpha=0.7)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['memory'], np.ones(window_size)/window_size, mode='valid')
        ax2.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax2.legend()
    ax2.set_title('Memory Usage Convergence')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Memory')
    ax2.grid(True, alpha=0.3)
    
    # 3. Latency convergence
    ax3 = axes[0, 2]
    ax3.plot(episodes, convergence_data['latency'], 'orange', linewidth=2, alpha=0.7)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['latency'], np.ones(window_size)/window_size, mode='valid')
        ax3.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax3.legend()
    ax3.set_title('Latency Convergence')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Latency (s)')
    ax3.grid(True, alpha=0.3)
    
    # 4. QoS convergence
    ax4 = axes[1, 0]
    ax4.plot(episodes, convergence_data['qos'], 'purple', linewidth=2, alpha=0.7)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['qos'], np.ones(window_size)/window_size, mode='valid')
        ax4.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax4.legend()
    ax4.set_title('QoS Convergence')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average QoS (BRISQUE)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Denoise steps convergence
    ax5 = axes[1, 1]
    ax5.plot(episodes, convergence_data['denoise_steps'], 'brown', linewidth=2, alpha=0.7)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['denoise_steps'], np.ones(window_size)/window_size, mode='valid')
        ax5.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax5.legend()
    ax5.set_title('Denoise Steps Convergence')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Average Denoise Steps')
    ax5.grid(True, alpha=0.3)
    
    # 6. Served users convergence
    ax6 = axes[1, 2]
    ax6.plot(episodes, convergence_data['served_users'], 'pink', linewidth=2, alpha=0.7)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['served_users'], np.ones(window_size)/window_size, mode='valid')
        ax6.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax6.legend()
    ax6.set_title('Served Users Convergence')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Number of Served Users')
    ax6.grid(True, alpha=0.3)
    
    # 7. Total system latency
    ax7 = axes[2, 0]
    ax7.plot(episodes, convergence_data['total_latency'], 'cyan', linewidth=2, alpha=0.7)
    if window_size > 1:
        moving_avg = np.convolve(convergence_data['total_latency'], np.ones(window_size)/window_size, mode='valid')
        ax7.plot(episodes[window_size-1:], moving_avg, 'r-', linewidth=3, label=f'MA({window_size})')
        ax7.legend()
    ax7.set_title('Total System Latency')
    ax7.set_xlabel('Episode')
    ax7.set_ylabel('Total Latency (s)')
    ax7.grid(True, alpha=0.3)
    
    # 8. Penalty and Bonus
    ax8 = axes[2, 1]
    ax8.plot(episodes, convergence_data['penalty'], 'red', linewidth=2, alpha=0.7, label='Penalty')
    ax8.plot(episodes, convergence_data['bonus'], 'green', linewidth=2, alpha=0.7, label='Bonus')
    ax8.set_title('Penalty and Bonus')
    ax8.set_xlabel('Episode')
    ax8.set_ylabel('Value')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Reward distribution
    ax9 = axes[2, 2]
    ax9.hist(convergence_data['rewards'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax9.axvline(np.mean(convergence_data['rewards']), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(convergence_data["rewards"]):.0f}')
    ax9.axvline(np.median(convergence_data['rewards']), color='orange', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(convergence_data["rewards"]):.0f}')
    ax9.set_title('Reward Distribution')
    ax9.set_xlabel('Reward')
    ax9.set_ylabel('Frequency')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{agent_name}_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Convergence plot for {agent_name} saved to: {save_dir}/{agent_name}_convergence.png")

def plot_comparative_convergence(all_convergence_data, save_dir='convergence_plots'):
    """Plot comparative convergence across all agents"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Define colors for each agent
    agent_colors = {
        'gaussian_ppo': '#1f77b4',
        'gaussian_a2c': '#ff7f0e', 
        'gaussian_dql': '#2ca02c',
        'a2c_diffusion': '#d62728',
        'bc_diffusion': '#9467bd',
        'ql_diffusion': '#8c564b'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparative Convergence Analysis - All Agents', fontsize=16, fontweight='bold')
    
    # 1. Reward convergence comparison
    ax1 = axes[0, 0]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        rewards = data['rewards']
        # Smooth the curve
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax1.plot(episodes[window_size-1:], smoothed, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax1.plot(episodes, rewards, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
    
    ax1.set_title('Reward Convergence Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory convergence comparison
    ax2 = axes[0, 1]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        memory = data['memory']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(memory, np.ones(window_size)/window_size, mode='valid')
            ax2.plot(episodes[window_size-1:], smoothed, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax2.plot(episodes, memory, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
    
    ax2.set_title('Memory Usage Convergence')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Memory')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Latency convergence comparison
    ax3 = axes[0, 2]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        latency = data['latency']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(latency, np.ones(window_size)/window_size, mode='valid')
            ax3.plot(episodes[window_size-1:], smoothed, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax3.plot(episodes, latency, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
    
    ax3.set_title('Latency Convergence')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Average Latency (s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. QoS convergence comparison
    ax4 = axes[1, 0]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        qos = data['qos']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(qos, np.ones(window_size)/window_size, mode='valid')
            ax4.plot(episodes[window_size-1:], smoothed, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax4.plot(episodes, qos, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
    
    ax4.set_title('QoS Convergence')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average QoS (BRISQUE)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Denoise steps convergence comparison
    ax5 = axes[1, 1]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        steps = data['denoise_steps']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
            ax5.plot(episodes[window_size-1:], smoothed, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax5.plot(episodes, steps, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
    
    ax5.set_title('Denoise Steps Convergence')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Average Denoise Steps')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Served users convergence comparison
    ax6 = axes[1, 2]
    for agent_name, data in all_convergence_data.items():
        episodes = data['episodes']
        served = data['served_users']
        window_size = max(1, len(episodes) // 20)
        if window_size > 1:
            smoothed = np.convolve(served, np.ones(window_size)/window_size, mode='valid')
            ax6.plot(episodes[window_size-1:], smoothed, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
        else:
            ax6.plot(episodes, served, 
                    color=agent_colors.get(agent_name, 'black'), 
                    linewidth=2, label=agent_name.replace('_', ' ').title())
    
    ax6.set_title('Served Users Convergence')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Number of Served Users')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/comparative_convergence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparative convergence plot saved to: {save_dir}/comparative_convergence.png")

def main():
    parser = argparse.ArgumentParser(description='Generate convergence plots for DRL agents')
    parser.add_argument('--agents', nargs='+', 
                       default=['gaussian_ppo', 'gaussian_a2c', 'gaussian_dql', 'a2c_diffusion', 'bc_diffusion', 'ql_diffusion'],
                       help='Agents to analyze')
    parser.add_argument('--num_users', type=int, default=10,
                       help='Number of users for convergence analysis')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes for convergence analysis')
    parser.add_argument('--max_steps', type=int, default=2000,
                       help='Maximum training steps')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='convergence_plots',
                       help='Directory to save plots')
    
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
    
    # Collect convergence data for all agents
    all_convergence_data = {}
    
    print(f"\n{'='*60}")
    print("COLLECTING CONVERGENCE DATA")
    print(f"{'='*60}")
    
    for agent_name in args.agents:
        try:
            convergence_data = collect_convergence_data(
                agent_name, args.num_users, device, 
                episodes=args.episodes, max_steps=args.max_steps
            )
            all_convergence_data[agent_name] = convergence_data
            
            # Plot individual convergence
            plot_convergence_metrics(convergence_data, agent_name, args.save_dir)
            
        except Exception as e:
            print(f"Error collecting data for {agent_name}: {e}")
            continue
    
    # Plot comparative convergence
    if all_convergence_data:
        print(f"\n{'='*60}")
        print("GENERATING COMPARATIVE PLOTS")
        print(f"{'='*60}")
        
        plot_comparative_convergence(all_convergence_data, args.save_dir)
        
        # Save raw convergence data
        with open(f'{args.save_dir}/convergence_data.json', 'w') as f:
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
            
            json.dump(convert_numpy(all_convergence_data), f, indent=2)
        
        print(f"\n{'='*60}")
        print("CONVERGENCE ANALYSIS COMPLETE!")
        print(f"{'='*60}")
        print(f"Generated plots in: {args.save_dir}/")
        print("- Individual agent convergence plots: {agent_name}_convergence.png")
        print("- Comparative convergence plot: comparative_convergence.png")
        print("- Raw data: convergence_data.json")
        
        # Print summary statistics
        print(f"\nSUMMARY STATISTICS:")
        for agent_name, data in all_convergence_data.items():
            final_reward = np.mean(data['rewards'][-10:]) if len(data['rewards']) >= 10 else np.mean(data['rewards'])
            final_memory = np.mean(data['memory'][-10:]) if len(data['memory']) >= 10 else np.mean(data['memory'])
            final_latency = np.mean(data['latency'][-10:]) if len(data['latency']) >= 10 else np.mean(data['latency'])
            final_qos = np.mean(data['qos'][-10:]) if len(data['qos']) >= 10 else np.mean(data['qos'])
            final_steps = np.mean(data['denoise_steps'][-10:]) if len(data['denoise_steps']) >= 10 else np.mean(data['denoise_steps'])
            
            print(f"{agent_name}:")
            print(f"  Final Reward: {final_reward:.2f}")
            print(f"  Final Memory: {final_memory:.2f}")
            print(f"  Final Latency: {final_latency:.4f}s")
            print(f"  Final QoS: {final_qos:.2f}")
            print(f"  Final Denoise Steps: {final_steps:.2f}")

if __name__ == "__main__":
    main()