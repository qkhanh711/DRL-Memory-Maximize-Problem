import json
import numpy as np
import os
import argparse


def print_statistics(log_dir, seed=42, num_mgu=50):
    """Print training statistics for all agents"""
    
    agents = ['gaussian_dql', 'gaussian_ppo', 'ppo_diffusion', 'ql_diffusion']
    agent_names = {
        'gaussian_dql': 'Gaussian DQL',
        'gaussian_ppo': 'Gaussian PPO',
        'ppo_diffusion': 'Diffusion PPO',
        'ql_diffusion': 'Diffusion QL'
    }
    
    print("\n" + "="*90)
    print(f"Training Results Summary (Seed={seed}, MGUs={num_mgu})")
    print("="*90)
    
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        
        if not os.path.exists(metrics_file):
            print(f"\n{agent_names[agent]}: No data found")
            continue
        
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        print(f"\n{agent_names[agent]}:")
        print("-" * 90)
        
        # Episode rewards statistics
        rewards = data['episode_rewards']
        if rewards:
            print(f"  Episodes trained:        {len(rewards)}")
            print(f"  Final episode reward:    {rewards[-1]:.2f}")
            print(f"  Average reward (last 20): {np.mean(rewards[-20:]):.2f} ± {np.std(rewards[-20:]):.2f}")
            print(f"  Average reward (all):    {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            print(f"  Max reward:              {max(rewards):.2f}")
            print(f"  Min reward:              {min(rewards):.2f}")
        
        # Evaluation rewards
        eval_rewards = data.get('eval_rewards', [])
        if eval_rewards:
            print(f"\n  Evaluation Performance:")
            print(f"    Best eval reward:      {max(eval_rewards):.2f}")
            print(f"    Final eval reward:     {eval_rewards[-1]:.2f}")
            print(f"    Average eval reward:   {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        
        # User distribution
        sat_users = data.get('avg_satellite_users', [])
        bs_users = data.get('avg_bs_users', [])
        if sat_users and bs_users:
            print(f"\n  User Distribution (final episodes):")
            print(f"    Satellite users:       {np.mean(sat_users[-20:]):.1f} / {num_mgu}")
            print(f"    BS users:              {np.mean(bs_users[-20:]):.1f} / {num_mgu}")
            print(f"    Sat/BS ratio:          {np.mean(sat_users[-20:])/np.mean(bs_users[-20:]):.2f}")
        
        # User utilities
        utilities = data.get('avg_user_utilities', [])
        if utilities:
            print(f"\n  User Utilities:")
            print(f"    Final avg utility:     {utilities[-1]:.2f}")
            print(f"    Max avg utility:       {max(utilities):.2f}")
    
    print("\n" + "="*90)
    
    # Comparison table
    print("\n" + "="*90)
    print("Performance Comparison")
    print("="*90)
    print(f"{'Agent':<20} {'Final Reward':<15} {'Best Eval':<15} {'Sat Users':<15} {'Avg Utility':<15}")
    print("-"*90)
    
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            rewards = data['episode_rewards']
            final_reward = np.mean(rewards[-20:]) if len(rewards) >= 20 else (rewards[-1] if rewards else 0)
            
            eval_rewards = data.get('eval_rewards', [])
            best_eval = max(eval_rewards) if eval_rewards else 0
            
            sat_users = data.get('avg_satellite_users', [])
            avg_sat = np.mean(sat_users[-20:]) if len(sat_users) >= 20 else (sat_users[-1] if sat_users else 0)
            
            utilities = data.get('avg_user_utilities', [])
            avg_util = utilities[-1] if utilities else 0
            
            print(f"{agent_names[agent]:<20} {final_reward:<15.2f} {best_eval:<15.2f} {avg_sat:<15.1f} {avg_util:<15.2f}")
    
    print("="*90 + "\n")
    
    # Print simple text-based reward progression
    print("\n" + "="*90)
    print("Reward Progression (every 10 episodes)")
    print("="*90)
    
    for agent in agents:
        metrics_file = os.path.join(log_dir, str(seed), agent, f'training_metrics_mgu_{num_mgu}.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            rewards = data['episode_rewards']
            if rewards:
                print(f"\n{agent_names[agent]}:")
                step = max(1, len(rewards) // 10)
                for i in range(0, len(rewards), step):
                    avg = np.mean(rewards[max(0, i-10):i+1])
                    bar_length = int(avg / 5)  # Scale for visualization
                    bar = '█' * max(0, bar_length)
                    print(f"  Episode {i:3d}: {avg:7.2f} {bar}")
    
    print("\n" + "="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Display Satellite-MEC training results')
    parser.add_argument('--log_dir', type=str, default='logs/satellite',
                       help='Directory containing training logs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used in training')
    parser.add_argument('--num_mgu', type=int, default=50, help='Number of MGUs')
    
    args = parser.parse_args()
    
    print_statistics(args.log_dir, args.seed, args.num_mgu)
    
    print("\nTo create plots, install matplotlib and run:")
    print("  pip install matplotlib")
    print("  python env/plot_satellite_results.py")


if __name__ == "__main__":
    main()
