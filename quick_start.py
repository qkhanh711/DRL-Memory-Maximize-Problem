#!/usr/bin/env python3
"""
Quick start script for DRL Memory Maximization Problem
This script demonstrates the complete workflow from training to evaluation
"""

import os
import sys
import json
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import Trainer, get_default_config
from evaluate import Evaluator


def quick_train_and_evaluate(agent_type='gaussian_a2c', episodes=200, eval_episodes=50):
    """Quick training and evaluation pipeline"""
    
    print("🚀 DRL Memory Maximization - Quick Start")
    print("="*50)
    
    # Step 1: Create configuration
    print(f"\n📋 Step 1: Setting up {agent_type} agent...")
    config = get_default_config()
    config['agent_type'] = agent_type
    config['total_episodes'] = episodes
    config['log_interval'] = max(1, episodes // 20)  # Log every 5% of episodes
    config['save_interval'] = max(1, episodes // 4)  # Save every 25% of episodes
    config['eval_interval'] = max(1, episodes // 10)  # Evaluate every 10% of episodes
    
    print(f"   • Episodes: {episodes}")
    print(f"   • Agent: {agent_type}")
    print(f"   • Device: {config['device']}")
    
    # Step 2: Train the agent
    print(f"\n🏋️ Step 2: Training {agent_type} agent...")
    trainer = Trainer(config)
    trainer.train()
    
    print("   ✅ Training completed!")
    
    # Step 3: Evaluate the trained agent
    print(f"\n📊 Step 3: Evaluating trained agent...")
    eval_config = config.copy()
    eval_config['model_path'] = "models/model_episode_final"
    
    evaluator = Evaluator(eval_config)
    stats, rewards, lengths, metrics = evaluator.evaluate(
        num_episodes=eval_episodes, 
        deterministic=True
    )
    
    # Step 4: Display results
    print(f"\n📈 Step 4: Results Summary")
    print("="*30)
    print(f"Average Reward: {stats['rewards']['mean']:.2f} ± {stats['rewards']['std']:.2f}")
    print(f"Average Length: {stats['lengths']['mean']:.1f} ± {stats['lengths']['std']:.1f}")
    
    if 'total_served' in stats:
        print(f"Users Served: {stats['total_served']['mean']:.1f} ± {stats['total_served']['std']:.1f}")
    if 'total_latency' in stats:
        print(f"Total Latency: {stats['total_latency']['mean']:.4f}s ± {stats['total_latency']['std']:.4f}s")
    if 'total_flops' in stats:
        print(f"Total FLOPS: {stats['total_flops']['mean']:.0f} ± {stats['total_flops']['std']:.0f}")
    if 'total_mem' in stats:
        print(f"Total Memory: {stats['total_mem']['mean']:.3f} ± {stats['total_mem']['std']:.3f}")
    
    # Step 5: Generate plots
    print(f"\n📊 Step 5: Generating evaluation plots...")
    evaluator.plot_results(stats, rewards, lengths, metrics, 'quick_start_results.png')
    print("   ✅ Plots saved as 'quick_start_results.png'")
    
    return stats, rewards, lengths, metrics


def compare_agents(agents=['gaussian_a2c', 'gaussian_ppo', 'gaussian_dql'], episodes=100):
    """Quick comparison of multiple agents"""
    
    print("🔬 Agent Comparison")
    print("="*30)
    
    results = {}
    
    for agent_type in agents:
        print(f"\n🏋️ Training {agent_type}...")
        
        # Train agent
        config = get_default_config()
        config['agent_type'] = agent_type
        config['total_episodes'] = episodes
        config['log_interval'] = max(1, episodes // 10)
        
        trainer = Trainer(config)
        trainer.train()
        
        # Evaluate agent
        eval_config = config.copy()
        eval_config['model_path'] = "models/model_episode_final"
        
        evaluator = Evaluator(eval_config)
        stats, rewards, lengths, metrics = evaluator.evaluate(num_episodes=20)
        
        results[agent_type] = {
            'avg_reward': stats['rewards']['mean'],
            'std_reward': stats['rewards']['std'],
            'avg_served': stats.get('total_served', {}).get('mean', 0),
            'avg_latency': stats.get('total_latency', {}).get('mean', 0)
        }
    
    # Display comparison
    print(f"\n📊 Comparison Results")
    print("="*50)
    print(f"{'Agent':<15} {'Avg Reward':<12} {'Std Reward':<12} {'Avg Served':<12} {'Avg Latency':<12}")
    print("-" * 70)
    
    for agent_type, result in results.items():
        print(f"{agent_type:<15} {result['avg_reward']:<12.2f} {result['std_reward']:<12.2f} "
              f"{result['avg_served']:<12.1f} {result['avg_latency']:<12.4f}")
    
    # Find best agent
    best_agent = max(results.items(), key=lambda x: x[1]['avg_reward'])
    print(f"\n🏆 Best performing agent: {best_agent[0]} (Reward: {best_agent[1]['avg_reward']:.2f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Quick start for DRL Memory Maximization')
    parser.add_argument('--mode', type=str, choices=['train', 'compare'], default='train',
                       help='Mode: train single agent or compare multiple agents')
    parser.add_argument('--agent', type=str, default='gaussian_a2c',
                       choices=['gaussian_a2c', 'gaussian_ppo', 'gaussian_dql', 'a2c_diffusion', 'ppo_diffusion', 'ql_diffusion'],
                       help='Agent type for training mode')
    parser.add_argument('--episodes', type=int, default=200,
                       help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=50,
                       help='Number of evaluation episodes')
    parser.add_argument('--agents', nargs='+', 
                       default=['gaussian_a2c', 'gaussian_ppo', 'gaussian_dql'],
                       help='Agents to compare (for compare mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Single agent training and evaluation
        quick_train_and_evaluate(args.agent, args.episodes, args.eval_episodes)
        
    elif args.mode == 'compare':
        # Multiple agent comparison
        compare_agents(args.agents, args.episodes)
    
    print(f"\n🎉 Quick start completed!")
    print(f"\nNext steps:")
    print(f"• Check the generated plots and logs")
    print(f"• Run 'python train.py --help' for more training options")
    print(f"• Run 'python evaluate.py --help' for more evaluation options")
    print(f"• Run 'python run_experiments.py --help' for systematic experiments")


if __name__ == '__main__':
    main()