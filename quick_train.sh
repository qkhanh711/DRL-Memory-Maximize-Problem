#!/bin/bash

# Quick training script for 4 models: DiffPPO, DiffQL, PPO, DQL
# Usage: bash quick_train.sh

echo "=============================================="
echo "Quick Training - 4 DRL Models"
echo "Models: DiffPPO, DiffQL, PPO, DQL"
echo "=============================================="
echo ""

# Fast configuration for testing
EPISODES=100
MAX_STEPS=10000
NUM_USERS=10
SEED=42

echo "Configuration:"
echo "  Episodes: $EPISODES"
echo "  Max Steps: $MAX_STEPS"
echo "  Num Users: $NUM_USERS"
echo "  Seed: $SEED"
echo ""

# Training with convergence
echo "Starting training..."
python convergence_analyze.py \
  --run_convergence \
  --agents ppo_diffusion ql_diffusion gaussian_ppo gaussian_dql \
  --num_users $NUM_USERS \
  --episodes $EPISODES \
  --max_steps $MAX_STEPS \
  --save_dir convergence_plots \
  --seed $SEED

# Move results
mkdir -p final_plot
if [ -d "convergence_plots" ]; then
  if [ -d "final_plot/convergence_plots" ]; then
    rm -rf "final_plot/convergence_plots"
  fi
  mv convergence_plots final_plot/
  echo ""
  echo "=============================================="
  echo "Training complete!"
  echo "Results saved to: final_plot/convergence_plots/"
  echo "=============================================="
fi

# Show results
echo ""
echo "Generated plots:"
ls -1 final_plot/convergence_plots/*.png 2>/dev/null || echo "No plots found"
