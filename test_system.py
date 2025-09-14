#!/usr/bin/env python3
"""
Test script to verify the DRL Memory Maximization system works correctly
"""

import os
import sys
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Test environment import
        from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
        print("✅ Environment imports successful")
        
        # Test agent imports
        from agents.gaussian_a2c import Gaussian_A2C
        from agents.gaussian_ppo import Gaussian_PPO
        from agents.gaussian_dql import Gaussian_DQL
        from agents.a2c_diffusion import Diffusion_A2C
        from agents.ppo_diffusion import Diffusion_PPO
        from agents.ql_diffusion import Diffusion_QL
        from agents.bc_diffusion import Diffusion_BC
        print("✅ Agent imports successful")
        
        # Test utility imports
        from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
        from utils.logger import Logger, MetricsTracker
        print("✅ Utility imports successful")
        
        # Test main scripts
        from train import Trainer, get_default_config
        from evaluate import Evaluator
        print("✅ Main script imports successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_environment():
    """Test environment creation and basic functionality"""
    print("\nTesting environment...")
    
    try:
        from env.m_env import GAIServiceEnv_v1, EnvConfig_v1
        
        # Create environment
        config = EnvConfig_v1("TestEnv")
        env = GAIServiceEnv_v1(config, seed=42)
        
        # Test reset
        state = env.reset()
        print(f"✅ Environment reset successful, state shape: {state.shape}")
        
        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"✅ Environment step successful, reward: {reward:.2f}")
        
        # Test episode
        state = env.reset()
        total_reward = 0
        for _ in range(5):  # Short episode
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                break
        
        print(f"✅ Episode completed, total reward: {total_reward:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        traceback.print_exc()
        return False


def test_agents():
    """Test agent creation and basic functionality"""
    print("\nTesting agents...")
    
    try:
        import torch
        from agents.gaussian_a2c import Gaussian_A2C
        
        # Test agent creation
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        agent = Gaussian_A2C(
            state_dim=60,  # 6 * 10 users
            action_dim=20,  # 2 * 10 users
            max_action=1.0,
            device=device
        )
        print("✅ Agent creation successful")
        
        # Test action sampling
        state = torch.randn(60)
        action = agent.sample_action(state.numpy())
        print(f"✅ Action sampling successful, action shape: {action.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        traceback.print_exc()
        return False


def test_replay_buffer():
    """Test replay buffer functionality"""
    print("\nTesting replay buffer...")
    
    try:
        from utils.replay_buffer import ReplayBuffer
        import torch
        
        # Create replay buffer
        buffer = ReplayBuffer(capacity=1000, device='cpu')
        
        # Add some transitions
        for _ in range(10):
            state = torch.randn(60)
            action = torch.randn(20)
            next_state = torch.randn(60)
            reward = torch.randn(1)
            done = torch.tensor(0.0)
            
            buffer.push(state.numpy(), action.numpy(), next_state.numpy(), 
                       reward.item(), done.item())
        
        print(f"✅ Replay buffer push successful, size: {len(buffer)}")
        
        # Test sampling
        batch = buffer.sample(5)
        print(f"✅ Replay buffer sampling successful, batch size: {len(batch[0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Replay buffer test failed: {e}")
        traceback.print_exc()
        return False


def test_training_pipeline():
    """Test basic training pipeline"""
    print("\nTesting training pipeline...")
    
    try:
        from train import Trainer, get_default_config
        
        # Create minimal config for testing
        config = get_default_config()
        config['agent_type'] = 'gaussian_a2c'
        config['total_episodes'] = 2  # Very short training
        config['max_steps_per_episode'] = 3
        config['min_buffer_size'] = 5
        config['train_iterations'] = 1
        config['batch_size'] = 4
        config['log_interval'] = 1
        config['save_interval'] = 1
        config['eval_interval'] = 1
        
        # Create trainer
        trainer = Trainer(config)
        print("✅ Trainer creation successful")
        
        # Run short training
        trainer.train()
        print("✅ Training pipeline successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation pipeline"""
    print("\nTesting evaluation pipeline...")
    
    try:
        from evaluate import Evaluator
        from train import get_default_config
        
        # Create config
        config = get_default_config()
        config['agent_type'] = 'gaussian_a2c'
        config['model_path'] = 'models/model_episode_final'
        
        # Create evaluator
        evaluator = Evaluator(config)
        print("✅ Evaluator creation successful")
        
        # Run short evaluation
        stats, rewards, lengths, metrics = evaluator.evaluate(num_episodes=2)
        print(f"✅ Evaluation successful, avg reward: {stats['rewards']['mean']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🧪 DRL Memory Maximization System Test")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("Environment", test_environment),
        ("Agents", test_agents),
        ("Replay Buffer", test_replay_buffer),
        ("Training Pipeline", test_training_pipeline),
        ("Evaluation", test_evaluation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print(f"{'='*50}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("• Run 'python quick_start.py' for a quick demo")
        print("• Run 'python train.py --help' for training options")
        print("• Run 'python evaluate.py --help' for evaluation options")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all dependencies are installed:")
        print("• pip install -r requirements.txt")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)