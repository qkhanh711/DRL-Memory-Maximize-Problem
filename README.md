# DRL Memory Maximization Problem

A comprehensive Deep Reinforcement Learning framework for solving memory optimization problems in AI service environments. This project implements multiple state-of-the-art RL algorithms and provides a complete training and evaluation pipeline.

## 🚀 Features

- **Multiple RL Algorithms**: A2C, BC, Gaussian A2C, Gaussian DQL, Gaussian PPO, PPO Diffusion, QL Diffusion
- **Flexible Environment**: Configurable AI service environment with user mobility and resource constraints
- **Comprehensive Training**: Full training pipeline with logging, monitoring, and model saving
- **Evaluation Tools**: Detailed evaluation and comparison capabilities
- **Configuration System**: Easy-to-use configuration management for different experiments

## 📁 Project Structure

```
DRL-Memory-Maximize-Problem/
├── env/
│   └── m_env.py                 # Main environment implementation
├── agents/
│   ├── a2c_diffusion.py         # A2C with Diffusion policy
│   ├── bc_diffusion.py          # Behavioral Cloning with Diffusion
│   ├── gaussian_a2c.py          # Gaussian A2C
│   ├── gaussian_dql.py          # Gaussian Deep Q-Learning
│   ├── gaussian_ppo.py          # Gaussian PPO
│   ├── ppo_diffusion.py         # PPO with Diffusion policy
│   └── ql_diffusion.py          # Q-Learning with Diffusion
├── utils/
│   ├── replay_buffer.py         # Replay buffer implementations
│   └── logger.py               # Logging utilities
├── configs/
│   ├── default_config.json     # Default configuration
│   └── agent_configs.py        # Agent-specific configurations
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation script
└── README.md                   # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd DRL-Memory-Maximize-Problem
```

2. **Install dependencies**:
```bash
pip install torch numpy matplotlib gym wandb tensorboard
```

3. **Verify installation**:
```bash
python train.py --help
```

## 🎯 Quick Start

### Basic Training

Train a Gaussian A2C agent with default settings:

```bash
python train.py --agent gaussian_a2c --episodes 1000
```

### Training with Custom Configuration

```bash
python train.py --config configs/default_config.json --agent gaussian_ppo --episodes 2000 --lr 0.001
```

### Evaluation

Evaluate a trained model:

```bash
python evaluate.py --config configs/default_config.json --model_path models/model_episode_1000 --episodes 100
```

### Compare Multiple Agents

```bash
python evaluate.py --config configs/default_config.json --model_path models/model_episode_1000 --compare gaussian_a2c gaussian_ppo gaussian_dql --episodes 50
```

## 📊 Supported Agents

| Agent | Type | Description | Best For |
|-------|------|-------------|----------|
| `gaussian_a2c` | On-policy | Gaussian Actor-Critic with continuous actions | Stable learning, continuous control |
| `gaussian_dql` | Off-policy | Deep Q-Learning with Gaussian policy | Sample efficiency, exploration |
| `gaussian_ppo` | On-policy | Proximal Policy Optimization with Gaussian policy | Stable learning, high-dimensional actions |
| `a2c_diffusion` | On-policy | A2C with Diffusion-based policy | Complex action distributions |
| `ppo_diffusion` | On-policy | PPO with Diffusion-based policy | High-quality action generation |
| `ql_diffusion` | Off-policy | Q-Learning with Diffusion-based policy | Complex state-action relationships |
| `bc_diffusion` | Imitation | Behavioral Cloning with Diffusion | Learning from demonstrations |

## ⚙️ Configuration

### Environment Configuration

The environment can be configured through the `EnvConfig_v1` function in `env/m_env.py`:

```python
config = {
    "num_users": 10,           # Number of users
    "T": 10,                   # Episode length
    "sys_tau": 4,              # System latency budget
    "Gmax": 5e9,               # FLOPS budget
    "Mmax": 48,                # Memory budget
    "qos_required": 30,        # QoS target
    # ... more parameters
}
```

### Agent Configuration

Each agent has specific hyperparameters. See `configs/agent_configs.py` for detailed configurations:

```python
# Example: Gaussian A2C configuration
gaussian_a2c_config = {
    "lr": 3e-4,
    "gamma": 0.99,
    "value_coef": 0.5,
    "ent_coef": 0.01,
    "grad_norm": 1.0
}
```

## 📈 Training Process

1. **Environment Reset**: Initialize users with random positions and requirements
2. **Action Selection**: Agent selects actions (serve/skip + denoise steps) for each user
3. **Environment Step**: Compute rewards based on QoS, latency, resource usage
4. **Experience Storage**: Store transitions in replay buffer
5. **Agent Training**: Update agent parameters using stored experiences
6. **Logging**: Record metrics and save models periodically

## 📊 Evaluation Metrics

- **Episode Reward**: Total reward accumulated per episode
- **Users Served**: Number of users successfully served
- **Latency**: Total system latency
- **Resource Usage**: FLOPS and memory consumption
- **QoS Performance**: Quality of service metrics
- **Constraint Violations**: Penalties for exceeding limits

## 🔧 Advanced Usage

### Custom Environment

Modify `env/m_env.py` to create custom environment configurations:

```python
def custom_env_config():
    config = EnvConfig_v1("CustomEnv")
    config["num_users"] = 20
    config["T"] = 15
    # ... custom parameters
    return config
```

### Custom Agent

Create a new agent by inheriting from the base agent class:

```python
class CustomAgent:
    def __init__(self, state_dim, action_dim, max_action, device, **kwargs):
        # Initialize your agent
        pass
    
    def train(self, replay_buffer, iterations, batch_size):
        # Implement training logic
        pass
    
    def sample_action(self, state, deterministic=False):
        # Implement action selection
        pass
```

### Hyperparameter Tuning

Use the configuration system to experiment with different hyperparameters:

```python
from configs.agent_configs import get_hyperparameter_ranges

# Get hyperparameter ranges for optimization
ranges = get_hyperparameter_ranges()
```

## 📝 Logging and Monitoring

### TensorBoard

Enable TensorBoard logging:

```bash
python train.py --agent gaussian_a2c --use_tensorboard
tensorboard --logdir logs
```

### Weights & Biases

Enable W&B logging:

```bash
python train.py --agent gaussian_a2c --wandb
```

### Custom Logging

Use the built-in logger for custom metrics:

```python
from utils.logger import Logger

logger = Logger(log_dir="logs", use_tensorboard=True)
logger.log_scalar("custom_metric", value, step)
```

## 🧪 Experimentation

### Running Multiple Experiments

```bash
# Train different agents
python train.py --agent gaussian_a2c --episodes 1000 --seed 42
python train.py --agent gaussian_ppo --episodes 1000 --seed 42
python train.py --agent gaussian_dql --episodes 1000 --seed 42

# Compare results
python evaluate.py --compare gaussian_a2c gaussian_ppo gaussian_dql --episodes 100
```

### Configuration Variants

```python
from configs.agent_configs import create_config_variants

# Create multiple configuration variants
variants = create_config_variants(base_config, "gaussian_a2c")
for variant in variants:
    # Run training with each variant
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Training Instability**: Lower learning rate or increase gradient clipping
3. **Poor Performance**: Try different agent types or hyperparameters
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 References

- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Deep Q-Learning](https://arxiv.org/abs/1312.5602)
- [Actor-Critic Methods](https://arxiv.org/abs/1602.01783)
- [Diffusion Models for RL](https://arxiv.org/abs/2208.06193)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gym for the environment framework
- PyTorch for deep learning capabilities
- The RL community for algorithm implementations