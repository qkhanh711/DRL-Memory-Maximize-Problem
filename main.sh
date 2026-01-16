#!/bin/bash

# Configuration for training 4 models: DiffPPO, DiffQL, PPO, DQL
EPISODES=1000
MAX_STEPS=100000
SEED=42

echo "============================================================"
echo "Training and Analysis Pipeline for 4 DRL Models"
echo "Models: DiffPPO, DiffQL, PPO, DQL"
echo "============================================================"
echo ""

# Quick start: Uncomment one of the following commands

# 1. Train models with convergence analysis (recommended)
bash run.sh train --episodes $EPISODES --max_steps $MAX_STEPS --seed $SEED

# 2. Performance analysis across different user counts
# bash run.sh analyze --episodes $EPISODES --max_steps $MAX_STEPS --seed $SEED

# 3. QoS target sweep analysis
# bash run.sh qos-sweep --seed $SEED

# 4. Complete pipeline (train + analyze + qos-sweep)
# bash run.sh full --seed $SEED

# 5. Replot from saved data
# bash run.sh replot-convergence
# bash run.sh replot-analyze