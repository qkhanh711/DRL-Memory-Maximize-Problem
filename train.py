import os
import json
import numpy as np
import torch
import argparse
from datetime import datetime
import random

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
# Simple logger for training metrics
class TrainingLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = {}
    
    def log_scalar(self, key, value, step):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append((step, value))
    
    def log(self, message):
        print(message)

def convert_numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    else:
        return obj

def set_seed(seed):
    """Set random seeds for reproducibility"""
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

def train_agent(agent_name, config, train_config, device):
    """Train a single agent"""
    print(f"\n{'='*50}")
    print(f"Training {agent_name}")
    print(f"{'='*50}")
    
    # Create environment
    env = GAIServiceEnv_v1(config, seed=train_config['seed'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    # Create agent
    agent = create_agent(agent_name, state_dim, action_dim, max_action, device, **train_config.get('agent_params', {}))
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=train_config['buffer_size'], device=device)
    
    # Create logger
    log_dir = f"logs/{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(log_dir, exist_ok=True)
    logger = TrainingLogger(log_dir)
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'training_losses': [],
        'eval_rewards': [],
        'eval_lengths': []
    }
    
    # Training loop
    total_steps = 0
    episode = 0
    
    while total_steps < train_config['max_steps']:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Episode loop
        while not done and episode_length < train_config['max_episode_length']:
            # Sample action
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
            if replay_buffer.size() >= train_config['min_buffer_size']:
                # Sample multiple batches for training
                for _ in range(train_config['train_frequency']):
                    if replay_buffer.size() >= train_config['batch_size']:
                        train_metrics = agent.train(
                            replay_buffer, 
                            iterations=1, 
                            batch_size=train_config['batch_size']
                        )
                        
                        # Log training metrics
                        for key, values in train_metrics.items():
                            if values:
                                logger.log_scalar(f"train/{key}", values[-1], total_steps)
        
        # Episode finished
        episode += 1
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Log episode metrics
        logger.log_scalar("episode/reward", episode_reward, episode)
        logger.log_scalar("episode/length", episode_length, episode)
        logger.log_scalar("episode/total_steps", total_steps, episode)
        
        # Store metrics
        training_metrics['episode_rewards'].append(episode_reward)
        training_metrics['episode_lengths'].append(episode_length)
        
        # Evaluation
        if episode % train_config['eval_frequency'] == 0:
            eval_reward, eval_length = evaluate_agent(agent, env, train_config['eval_episodes'])
            training_metrics['eval_rewards'].append(eval_reward)
            training_metrics['eval_lengths'].append(eval_length)
            
            logger.log_scalar("eval/reward", eval_reward, episode)
            logger.log_scalar("eval/length", eval_length, episode)
            
            print(f"Episode {episode}, Eval Reward: {eval_reward:.2f}, Eval Length: {eval_length:.2f}")
        
        # Print progress
        if episode % train_config['print_frequency'] == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths[-100:]) if len(episode_lengths) >= 100 else np.mean(episode_lengths)
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}, Total Steps: {total_steps}")
        
        # Save model
        if episode % train_config['save_frequency'] == 0:
            agent.save_model(log_dir, id=episode)
    
    # Save final metrics
    training_metrics['config'] = convert_numpy_to_list(config)
    training_metrics['train_config'] = convert_numpy_to_list(train_config)
    training_metrics['agent_name'] = agent_name
    
    with open(f"{log_dir}/training_metrics.json", 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    # Save final model
    agent.save_model(log_dir)
    
    print(f"Training completed for {agent_name}")
    print(f"Final average reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Logs saved to: {log_dir}")
    
    return training_metrics, log_dir

def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance"""
    total_rewards = []
    total_lengths = []
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 1000:  # Max episode length for eval
            # Try deterministic=True first, fallback to regular call if not supported
            try:
                action = agent.sample_action(state, deterministic=True)
            except TypeError:
                action = agent.sample_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
    
    return np.mean(total_rewards), np.mean(total_lengths)

def main():
    parser = argparse.ArgumentParser(description='Train DRL agents on GAIServiceEnv')
    parser.add_argument('--agent', type=str, default='all', 
                       choices=['all', 'a2c_diffusion', 'bc_diffusion', 'gaussian_a2c', 
                               'gaussian_dql', 'gaussian_ppo', 'ppo_diffusion', 'ql_diffusion'],
                       help='Agent to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_steps', type=int, default=100000, help='Maximum training steps')
    parser.add_argument('--max_episode_length', type=int, default=1000, help='Maximum episode length')
    parser.add_argument('--buffer_size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--min_buffer_size', type=int, default=1000, help='Minimum buffer size before training')
    parser.add_argument('--train_frequency', type=int, default=1, help='Training frequency (steps)')
    parser.add_argument('--eval_frequency', type=int, default=50, help='Evaluation frequency (episodes)')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of evaluation episodes')
    parser.add_argument('--print_frequency', type=int, default=10, help='Print frequency (episodes)')
    parser.add_argument('--save_frequency', type=int, default=100, help='Save frequency (episodes)')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Environment configuration
    config = EnvConfig_v1("GAIServiceEnv")
    
    # Training configuration
    train_config = {
        'seed': args.seed,
        'max_steps': args.max_steps,
        'max_episode_length': args.max_episode_length,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'min_buffer_size': args.min_buffer_size,
        'train_frequency': args.train_frequency,
        'eval_frequency': args.eval_frequency,
        'eval_episodes': args.eval_episodes,
        'print_frequency': args.print_frequency,
        'save_frequency': args.save_frequency,
        'agent_params': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'discount': 0.99,
            'grad_norm': 1.0,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.5,
        }
    }
    
    # Agent-specific parameters
    agent_configs = {
        'a2c_diffusion': {
            'agent_params': {
                'lr': 1e-4,
                'gamma': 0.99,
                'n_timesteps': 20,
                'ema_decay': 0.995,
                'entropy_coef': 0.01,
                'value_loss_coef': 0.25,
                'grad_norm': 0.25,
            }
        },
        'bc_diffusion': {
            'agent_params': {
                'lr': 2e-4,
                'discount': 0.99,
                'tau': 0.005,
                'n_timesteps': 100,
            }
        },
        'gaussian_a2c': {
            'agent_params': {
                'lr': 3e-4,
                'gamma': 0.99,
                'value_coef': 0.5,
                'ent_coef': 0.01,
                'grad_norm': 1.0,
            }
        },
        'gaussian_dql': {
            'agent_params': {
                'discount': 0.99,
                'tau': 0.005,
                'lr': 3e-4,
                'grad_norm': 1.0,
                'noise_scale': 0.3,
                'noise_type': 'gaussian',
                'epsilon': 0.01,
                'policy_delay': 2,
            }
        },
        'gaussian_ppo': {
            'agent_params': {
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
        },
        'ppo_diffusion': {
            'agent_params': {
                'gamma': 0.99,
                'tau': 0.95,
                'clip_param': 0.2,
                'n_timesteps': 10,
                'ema_decay': 0.99,
                'lr': 2e-4,
                'entropy_coef': 0.02,
                'value_loss_coef': 0.25,
                'grad_norm': 1.0,
            }
        },
        'ql_diffusion': {
            'agent_params': {
                'discount': 0.99,
                'tau': 0.005,
                'n_timesteps': 100,
                'ema_decay': 0.995,
                'lr': 3e-4,
                'grad_norm': 1.0,
                'eta': 1.0,
            }
        }
    }
    
    # Determine which agents to train
    if args.agent == 'all':
        agents_to_train = ['a2c_diffusion', 'bc_diffusion', 'gaussian_a2c', 
                          'gaussian_dql', 'gaussian_ppo', 'ppo_diffusion', 'ql_diffusion']
    else:
        agents_to_train = [args.agent]
    
    # Train agents
    all_results = {}
    
    for agent_name in agents_to_train:
        try:
            # Update train_config with agent-specific parameters
            current_train_config = train_config.copy()
            if agent_name in agent_configs:
                current_train_config.update(agent_configs[agent_name])
            
            # Train agent
            metrics, log_dir = train_agent(agent_name, config, current_train_config, device)
            all_results[agent_name] = {
                'metrics': metrics,
                'log_dir': log_dir
            }
            
        except Exception as e:
            print(f"Error training {agent_name}: {e}")
            continue
    
    # Save summary results
    summary = {
        'config': convert_numpy_to_list(config),
        'train_config': convert_numpy_to_list(train_config),
        'results': convert_numpy_to_list(all_results),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Training Summary")
    print(f"{'='*50}")
    for agent_name, result in all_results.items():
        if 'metrics' in result:
            final_reward = np.mean(result['metrics']['episode_rewards'][-100:]) if result['metrics']['episode_rewards'] else 0
            print(f"{agent_name}: Final Avg Reward = {final_reward:.2f}")
    print(f"Summary saved to: training_summary.json")

if __name__ == "__main__":
    main()