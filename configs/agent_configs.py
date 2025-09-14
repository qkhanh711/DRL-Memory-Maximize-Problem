"""
Agent-specific configuration presets
"""

def get_agent_config(agent_type):
    """Get configuration for specific agent type"""
    
    configs = {
        'a2c_diffusion': {
            'lr': 1e-4,
            'gamma': 0.99,
            'beta_schedule': 'linear',
            'n_timesteps': 20,
            'ema_decay': 0.995,
            'step_start_ema': 2000,
            'update_ema_every': 10,
            'grad_norm': 0.25,
            'entropy_coef': 0.01,
            'value_loss_coef': 0.25,
            'lr_decay': False,
            'max_grad_norm': 0.5,
            'total_episodes': 2000,
            'train_iterations': 5
        },
        
        'bc_diffusion': {
            'lr': 2e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'beta_schedule': 'linear',
            'n_timesteps': 100,
            'total_episodes': 1500,
            'train_iterations': 20
        },
        
        'gaussian_a2c': {
            'lr': 3e-4,
            'gamma': 0.99,
            'value_coef': 0.5,
            'ent_coef': 0.01,
            'grad_norm': 1.0,
            'total_episodes': 1500,
            'train_iterations': 10
        },
        
        'gaussian_dql': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'grad_norm': 1.0,
            'noise_scale': 0.3,
            'noise_type': 'gaussian',
            'epsilon': 0.01,
            'policy_delay': 2,
            'total_episodes': 2000,
            'train_iterations': 10
        },
        
        'gaussian_ppo': {
            'lr': 7e-3,
            'gamma': 0.99,
            'tau': 0.005,
            'grad_norm': 1.0,
            'clip_ratio': 0.2,
            'value_clip_ratio': 0.2,
            'norm_adv': True,
            'horizon_steps': 1,
            'ent_coef': 0.01,
            'noise_scale': 0.1,
            'noise_type': 'gaussian',
            'epsilon': 0.1,
            'lr_decay': False,
            'lr_maxt': 1000,
            'total_episodes': 2000,
            'train_iterations': 10
        },
        
        'ppo_diffusion': {
            'lr': 2e-4,
            'gamma': 0.99,
            'tau': 0.95,
            'clip_param': 0.2,
            'beta_schedule': 'linear',
            'n_timesteps': 10,
            'ema_decay': 0.99,
            'step_start_ema': 500,
            'update_ema_every': 3,
            'grad_norm': 1.0,
            'entropy_coef': 0.02,
            'value_loss_coef': 0.25,
            'warmup_steps': 200,
            'lr_decay': True,
            'lr_maxt': 10000,
            'total_episodes': 2000,
            'train_iterations': 5
        },
        
        'ql_diffusion': {
            'lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            'max_q_backup': False,
            'eta': 1.0,
            'beta_schedule': 'linear',
            'n_timesteps': 100,
            'ema_decay': 0.995,
            'step_start_ema': 1000,
            'update_ema_every': 5,
            'grad_norm': 1.0,
            'lr_decay': False,
            'lr_maxt': 1000,
            'total_episodes': 2000,
            'train_iterations': 10
        }
    }
    
    return configs.get(agent_type, {})


def get_hyperparameter_ranges():
    """Get hyperparameter ranges for hyperparameter optimization"""
    
    return {
        'lr': [1e-5, 1e-2],
        'gamma': [0.9, 0.999],
        'batch_size': [32, 256],
        'buffer_size': [10000, 500000],
        'grad_norm': [0.1, 2.0],
        'entropy_coef': [0.001, 0.1],
        'value_loss_coef': [0.1, 1.0],
        'noise_scale': [0.01, 1.0],
        'epsilon': [0.001, 0.3],
        'clip_ratio': [0.1, 0.5],
        'tau': [0.001, 0.1],
        'n_timesteps': [5, 50],
        'ema_decay': [0.9, 0.999]
    }


def get_environment_configs():
    """Get different environment configurations for testing"""
    
    return {
        'easy': {
            'num_users': 5,
            'T': 5,
            'sys_tau': 6,
            'Gmax': 1e10,
            'Mmax': 100
        },
        'medium': {
            'num_users': 10,
            'T': 10,
            'sys_tau': 4,
            'Gmax': 5e9,
            'Mmax': 48
        },
        'hard': {
            'num_users': 20,
            'T': 15,
            'sys_tau': 3,
            'Gmax': 2e9,
            'Mmax': 30
        }
    }


def create_config_variants(base_config, agent_type):
    """Create multiple configuration variants for an agent"""
    
    agent_config = get_agent_config(agent_type)
    variants = []
    
    # Learning rate variants
    for lr in [1e-4, 3e-4, 1e-3]:
        variant = base_config.copy()
        variant.update(agent_config)
        variant['lr'] = lr
        variant['run_name'] = f"{agent_type}_lr_{lr}"
        variants.append(variant)
    
    # Batch size variants
    for batch_size in [32, 64, 128]:
        variant = base_config.copy()
        variant.update(agent_config)
        variant['batch_size'] = batch_size
        variant['run_name'] = f"{agent_type}_batch_{batch_size}"
        variants.append(variant)
    
    # Noise scale variants (for agents that support it)
    if agent_type in ['gaussian_dql', 'gaussian_ppo']:
        for noise_scale in [0.1, 0.3, 0.5]:
            variant = base_config.copy()
            variant.update(agent_config)
            variant['noise_scale'] = noise_scale
            variant['run_name'] = f"{agent_type}_noise_{noise_scale}"
            variants.append(variant)
    
    return variants