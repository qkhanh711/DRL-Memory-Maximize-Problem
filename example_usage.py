#!/usr/bin/env python3
"""
Example usage of the DRL Memory Maximization training system
"""

import os
import sys
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import Trainer, get_default_config
from evaluate import Evaluator


def example_basic_training():
    """Example: Basic training with Gaussian A2C"""
    print("="*60)
    print("EXAMPLE 1: Basic Training with Gaussian A2C")
    print("="*60)
    
    # Create configuration
    config = get_default_config()
    config['agent_type'] = 'gaussian_a2c'
    config['total_episodes'] = 100  # Short training for demo
    config['log_interval'] = 10
    config['save_interval'] = 50
    config['eval_interval'] = 25
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train the agent
    trainer.train()
    
    print("Training completed!")
    return trainer


def example_agent_comparison():
    """Example: Compare different agents"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Agent Comparison")
    print("="*60)
    
    agents = ['gaussian_a2c', 'gaussian_ppo', 'gaussian_dql']
    results = {}
    
    for agent_type in agents:
        print(f"\nTraining {agent_type}...")
        
        # Create configuration for this agent
        config = get_default_config()
        config['agent_type'] = agent_type
        config['total_episodes'] = 50  # Short training for demo
        config['log_interval'] = 10
        
        # Train agent
        trainer = Trainer(config)
        trainer.train()
        
        # Evaluate agent
        eval_config = config.copy()
        eval_config['model_path'] = f"models/model_episode_final"
        
        evaluator = Evaluator(eval_config)
        stats, rewards, lengths, metrics = evaluator.evaluate(num_episodes=10)
        
        results[agent_type] = {
            'avg_reward': stats['rewards']['mean'],
            'std_reward': stats['rewards']['std'],
            'avg_served': stats.get('total_served', {}).get('mean', 0)
        }
    
    # Print comparison results
    print("\n" + "="*40)
    print("COMPARISON RESULTS")
    print("="*40)
    print(f"{'Agent':<15} {'Avg Reward':<12} {'Std Reward':<12} {'Avg Served':<12}")
    print("-" * 50)
    
    for agent_type, result in results.items():
        print(f"{agent_type:<15} {result['avg_reward']:<12.2f} {result['std_reward']:<12.2f} {result['avg_served']:<12.2f}")
    
    return results


def example_custom_environment():
    """Example: Custom environment configuration"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Environment Configuration")
    print("="*60)
    
    # Import environment modules
    from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
    
    # Create custom environment configuration
    def custom_env_config():
        config = EnvConfig_v1("CustomEnv")
        # Make environment easier for demo
        config["num_users"] = 5
        config["T"] = 5
        config["sys_tau"] = 6  # More lenient latency constraint
        config["Gmax"] = 1e10  # Higher FLOPS budget
        config["Mmax"] = 100   # Higher memory budget
        return config
    
    # Test custom environment
    env_config = custom_env_config()
    env = GAIServiceEnv_v1(env_config, seed=42)
    
    print("Custom environment created with:")
    print(f"  Users: {env_config['num_users']}")
    print(f"  Episode length: {env_config['T']}")
    print(f"  Latency budget: {env_config['sys_tau']}s")
    print(f"  FLOPS budget: {env_config['Gmax']:.0e}")
    print(f"  Memory budget: {env_config['Mmax']}")
    
    # Run a few episodes
    for episode in range(3):
        state = env.reset()
        total_reward = 0
        
        for step in range(env_config['T']):
            action = env.action_space.sample()  # Random action
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode+1}: Reward = {total_reward:.2f}, Served = {info.get('total_served', 0)}")


def example_hyperparameter_tuning():
    """Example: Hyperparameter tuning"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Hyperparameter Tuning")
    print("="*60)
    
    # Define parameter ranges to test
    param_ranges = {
        'lr': [1e-4, 3e-4, 1e-3],
        'batch_size': [32, 64],
        'gamma': [0.95, 0.99]
    }
    
    best_config = None
    best_reward = -float('inf')
    results = []
    
    # Test different parameter combinations
    for lr in param_ranges['lr']:
        for batch_size in param_ranges['batch_size']:
            for gamma in param_ranges['gamma']:
                print(f"\nTesting: lr={lr}, batch_size={batch_size}, gamma={gamma}")
                
                # Create configuration
                config = get_default_config()
                config['agent_type'] = 'gaussian_a2c'
                config['lr'] = lr
                config['batch_size'] = batch_size
                config['gamma'] = gamma
                config['total_episodes'] = 30  # Short training for demo
                config['log_interval'] = 15
                
                # Train agent
                trainer = Trainer(config)
                trainer.train()
                
                # Evaluate agent
                eval_config = config.copy()
                eval_config['model_path'] = "models/model_episode_final"
                
                evaluator = Evaluator(eval_config)
                stats, rewards, lengths, metrics = evaluator.evaluate(num_episodes=5)
                
                avg_reward = stats['rewards']['mean']
                results.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'gamma': gamma,
                    'avg_reward': avg_reward
                })
                
                print(f"  Average reward: {avg_reward:.2f}")
                
                # Track best configuration
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_config = {
                        'lr': lr,
                        'batch_size': batch_size,
                        'gamma': gamma
                    }
    
    # Print results
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*50)
    print(f"{'LR':<8} {'Batch':<6} {'Gamma':<6} {'Avg Reward':<12}")
    print("-" * 35)
    
    for result in sorted(results, key=lambda x: x['avg_reward'], reverse=True):
        print(f"{result['lr']:<8.0e} {result['batch_size']:<6} {result['gamma']:<6} {result['avg_reward']:<12.2f}")
    
    print(f"\nBest configuration: {best_config}")
    print(f"Best average reward: {best_reward:.2f}")
    
    return results, best_config


def example_evaluation_analysis():
    """Example: Detailed evaluation and analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Detailed Evaluation and Analysis")
    print("="*60)
    
    # First train an agent
    config = get_default_config()
    config['agent_type'] = 'gaussian_ppo'
    config['total_episodes'] = 100
    config['log_interval'] = 20
    
    print("Training Gaussian PPO agent...")
    trainer = Trainer(config)
    trainer.train()
    
    # Evaluate with different settings
    eval_config = config.copy()
    eval_config['model_path'] = "models/model_episode_final"
    
    evaluator = Evaluator(eval_config)
    
    print("\nEvaluating agent...")
    
    # Deterministic evaluation
    print("\nDeterministic evaluation:")
    stats_det, rewards_det, lengths_det, metrics_det = evaluator.evaluate(
        num_episodes=20, deterministic=True
    )
    
    # Stochastic evaluation
    print("\nStochastic evaluation:")
    stats_stoch, rewards_stoch, lengths_stoch, metrics_stoch = evaluator.evaluate(
        num_episodes=20, deterministic=False
    )
    
    # Compare results
    print("\n" + "="*40)
    print("EVALUATION COMPARISON")
    print("="*40)
    print(f"{'Metric':<20} {'Deterministic':<15} {'Stochastic':<15}")
    print("-" * 50)
    print(f"{'Avg Reward':<20} {stats_det['rewards']['mean']:<15.2f} {stats_stoch['rewards']['mean']:<15.2f}")
    print(f"{'Std Reward':<20} {stats_det['rewards']['std']:<15.2f} {stats_stoch['rewards']['std']:<15.2f}")
    print(f"{'Avg Length':<20} {stats_det['lengths']['mean']:<15.2f} {stats_stoch['lengths']['mean']:<15.2f}")
    
    if 'total_served' in stats_det:
        print(f"{'Avg Served':<20} {stats_det['total_served']['mean']:<15.2f} {stats_stoch['total_served']['mean']:<15.2f}")
    
    # Plot results
    print("\nGenerating plots...")
    evaluator.plot_results(stats_det, rewards_det, lengths_det, metrics_det, 
                          'example_deterministic_eval.png')
    evaluator.plot_results(stats_stoch, rewards_stoch, lengths_stoch, metrics_stoch,
                          'example_stochastic_eval.png')
    
    print("Plots saved as 'example_deterministic_eval.png' and 'example_stochastic_eval.png'")


def main():
    """Run all examples"""
    print("DRL Memory Maximization - Example Usage")
    print("="*60)
    
    try:
        # Example 1: Basic training
        example_basic_training()
        
        # Example 2: Agent comparison
        example_agent_comparison()
        
        # Example 3: Custom environment
        example_custom_environment()
        
        # Example 4: Hyperparameter tuning
        example_hyperparameter_tuning()
        
        # Example 5: Evaluation analysis
        example_evaluation_analysis()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check the generated plots and logs")
        print("2. Try different agent types and configurations")
        print("3. Run longer training sessions for better results")
        print("4. Use the run_experiments.py script for systematic experiments")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()