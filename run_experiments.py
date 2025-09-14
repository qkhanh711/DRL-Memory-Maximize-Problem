#!/usr/bin/env python3
"""
Script to run multiple experiments with different agents and configurations
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
import itertools

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.agent_configs import get_agent_config, create_config_variants


def run_experiment(agent_type, config, experiment_name, base_dir="experiments"):
    """Run a single experiment"""
    
    # Create experiment directory
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Update config for this experiment
    config['save_dir'] = os.path.join(exp_dir, 'models')
    config['log_dir'] = os.path.join(exp_dir, 'logs')
    config['run_name'] = experiment_name
    
    # Save config
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run training
    cmd = [
        'python', 'train.py',
        '--config', config_path,
        '--agent', agent_type
    ]
    
    print(f"Running experiment: {experiment_name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
        if result.returncode == 0:
            print(f"✅ Experiment {experiment_name} completed successfully")
            return True
        else:
            print(f"❌ Experiment {experiment_name} failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {experiment_name} timed out")
        return False
    except Exception as e:
        print(f"💥 Experiment {experiment_name} failed with error: {e}")
        return False


def run_agent_comparison(agents, base_config, num_episodes=1000):
    """Run comparison between multiple agents"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/agent_comparison_{timestamp}"
    
    results = {}
    
    for agent_type in agents:
        print(f"\n{'='*50}")
        print(f"Training {agent_type}")
        print(f"{'='*50}")
        
        # Get agent-specific config
        agent_config = get_agent_config(agent_type)
        config = base_config.copy()
        config.update(agent_config)
        config['total_episodes'] = num_episodes
        
        # Run experiment
        experiment_name = f"{agent_type}_{timestamp}"
        success = run_experiment(agent_type, config, experiment_name, base_dir)
        
        results[agent_type] = {
            'success': success,
            'experiment_name': experiment_name,
            'config': config
        }
    
    # Save comparison results
    results_path = os.path.join(base_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("COMPARISON COMPLETE")
    print(f"{'='*50}")
    
    successful_agents = [agent for agent, result in results.items() if result['success']]
    failed_agents = [agent for agent, result in results.items() if not result['success']]
    
    print(f"✅ Successful agents: {successful_agents}")
    if failed_agents:
        print(f"❌ Failed agents: {failed_agents}")
    
    return results


def run_hyperparameter_sweep(agent_type, base_config, param_ranges, num_episodes=500):
    """Run hyperparameter sweep for a single agent"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/hyperparameter_sweep_{agent_type}_{timestamp}"
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"Running hyperparameter sweep for {agent_type}")
    print(f"Total combinations: {len(combinations)}")
    
    results = {}
    
    for i, combination in enumerate(combinations):
        # Create config with this parameter combination
        config = base_config.copy()
        agent_config = get_agent_config(agent_type)
        config.update(agent_config)
        config['total_episodes'] = num_episodes
        
        # Update parameters
        for param_name, param_value in zip(param_names, combination):
            config[param_name] = param_value
        
        # Create experiment name
        param_str = "_".join([f"{name}_{value}" for name, value in zip(param_names, combination)])
        experiment_name = f"{agent_type}_{param_str}_{timestamp}"
        
        print(f"\n--- Combination {i+1}/{len(combinations)} ---")
        print(f"Parameters: {dict(zip(param_names, combination))}")
        
        # Run experiment
        success = run_experiment(agent_type, config, experiment_name, base_dir)
        
        results[experiment_name] = {
            'success': success,
            'parameters': dict(zip(param_names, combination)),
            'config': config
        }
    
    # Save sweep results
    results_path = os.path.join(base_dir, 'sweep_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("HYPERPARAMETER SWEEP COMPLETE")
    print(f"{'='*50}")
    
    successful_runs = [name for name, result in results.items() if result['success']]
    failed_runs = [name for name, result in results.items() if not result['success']]
    
    print(f"✅ Successful runs: {len(successful_runs)}")
    print(f"❌ Failed runs: {len(failed_runs)}")
    
    return results


def run_environment_comparison(agent_type, base_config, env_configs, num_episodes=1000):
    """Run comparison across different environment configurations"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"experiments/env_comparison_{agent_type}_{timestamp}"
    
    results = {}
    
    for env_name, env_config in env_configs.items():
        print(f"\n{'='*50}")
        print(f"Testing {agent_type} on {env_name} environment")
        print(f"{'='*50}")
        
        # Update config with environment settings
        config = base_config.copy()
        agent_config = get_agent_config(agent_type)
        config.update(agent_config)
        config['total_episodes'] = num_episodes
        
        # Update environment parameters
        config.update(env_config)
        
        # Run experiment
        experiment_name = f"{agent_type}_{env_name}_{timestamp}"
        success = run_experiment(agent_type, config, experiment_name, base_dir)
        
        results[env_name] = {
            'success': success,
            'experiment_name': experiment_name,
            'config': config
        }
    
    # Save environment comparison results
    results_path = os.path.join(base_dir, 'env_comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*50}")
    print("ENVIRONMENT COMPARISON COMPLETE")
    print(f"{'='*50}")
    
    successful_envs = [env for env, result in results.items() if result['success']]
    failed_envs = [env for env, result in results.items() if not result['success']]
    
    print(f"✅ Successful environments: {successful_envs}")
    if failed_envs:
        print(f"❌ Failed environments: {failed_envs}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run multiple experiments')
    parser.add_argument('--experiment_type', type=str, 
                       choices=['agent_comparison', 'hyperparameter_sweep', 'env_comparison'],
                       required=True, help='Type of experiment to run')
    parser.add_argument('--agents', nargs='+', 
                       default=['gaussian_a2c', 'gaussian_ppo', 'gaussian_dql'],
                       help='Agents to compare (for agent_comparison)')
    parser.add_argument('--agent', type=str, 
                       default='gaussian_a2c',
                       help='Agent type (for hyperparameter_sweep and env_comparison)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes per experiment')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                       help='Base configuration file')
    parser.add_argument('--param_ranges', type=str,
                       help='JSON file with parameter ranges for hyperparameter sweep')
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = json.load(f)
    
    if args.experiment_type == 'agent_comparison':
        print("Running agent comparison experiment...")
        results = run_agent_comparison(args.agents, base_config, args.episodes)
        
    elif args.experiment_type == 'hyperparameter_sweep':
        if not args.param_ranges:
            # Default parameter ranges
            param_ranges = {
                'lr': [1e-4, 3e-4, 1e-3],
                'batch_size': [32, 64, 128],
                'gamma': [0.95, 0.99, 0.995]
            }
        else:
            with open(args.param_ranges, 'r') as f:
                param_ranges = json.load(f)
        
        print("Running hyperparameter sweep experiment...")
        results = run_hyperparameter_sweep(args.agent, base_config, param_ranges, args.episodes)
        
    elif args.experiment_type == 'env_comparison':
        # Default environment configurations
        from configs.agent_configs import get_environment_configs
        env_configs = get_environment_configs()
        
        print("Running environment comparison experiment...")
        results = run_environment_comparison(args.agent, base_config, env_configs, args.episodes)
    
    print(f"\n🎉 All experiments completed!")
    print(f"Results saved in experiments/ directory")


if __name__ == '__main__':
    main()