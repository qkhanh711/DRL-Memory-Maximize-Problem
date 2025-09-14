#!/bin/bash

# ============================================================
# CONFIGURATION PARAMETERS
# ============================================================
NUM_EPISODES=500
NUM_STEPS=10
SEED=42

# Training parameters
MAX_TRAINING_STEPS=50000
MAX_EPISODE_LENGTH=10
EVAL_FREQUENCY=25
PRINT_FREQUENCY=5
SAVE_FREQUENCY=50

# Analysis parameters
USER_COUNTS="8 10 12"
ANALYSIS_EPISODES=20
ANALYSIS_STEPS=300
CONVERGENCE_EPISODES=50
CONVERGENCE_STEPS=1000

echo "============================================================"
echo "DRL Memory Maximize Problem - Complete Analysis Pipeline"
echo "============================================================"
echo "Configuration:"
echo "  NUM_EPISODES: $NUM_EPISODES"
echo "  NUM_STEPS: $NUM_STEPS"
echo "  SEED: $SEED"
echo "  MAX_TRAINING_STEPS: $MAX_TRAINING_STEPS"
echo "  USER_COUNTS: $USER_COUNTS"
echo "============================================================"

# Create directories
mkdir -p final_plot
mkdir -p analysis_plots
mkdir -p convergence_plots
mkdir -p logs

# Kill any existing tmux sessions with these names
tmux kill-session -t train_ppo_diffusion 2>/dev/null || true
tmux kill-session -t train_gaussian_ppo 2>/dev/null || true
tmux kill-session -t train_ql_diffusion 2>/dev/null || true
tmux kill-session -t train_gaussian_dql 2>/dev/null || true

echo ""
echo "Step 1: Starting parallel training on 2 GPUs..."
echo "============================================================"
echo "GPU 0: PPO Diffusion + Gaussian PPO"
echo "GPU 1: QL Diffusion + Gaussian DQL"
echo ""

# Start training sessions
echo "Starting PPO Diffusion training on CUDA:0..."
tmux new-session -d -s train_ppo_diffusion "python train.py --agent ppo_diffusion --device cuda:0 --max_steps $MAX_TRAINING_STEPS --max_episode_length $MAX_EPISODE_LENGTH --eval_frequency $EVAL_FREQUENCY --print_frequency $PRINT_FREQUENCY --save_frequency $SAVE_FREQUENCY --seed $SEED"

echo "Starting Gaussian PPO training on CUDA:0..."
tmux new-session -d -s train_gaussian_ppo "python train.py --agent gaussian_ppo --device cuda:0 --max_steps $MAX_TRAINING_STEPS --max_episode_length $MAX_EPISODE_LENGTH --eval_frequency $EVAL_FREQUENCY --print_frequency $PRINT_FREQUENCY --save_frequency $SAVE_FREQUENCY --seed $SEED"

echo "Starting QL Diffusion training on CUDA:1..."
tmux new-session -d -s train_ql_diffusion "python train.py --agent ql_diffusion --device cuda:1 --max_steps $MAX_TRAINING_STEPS --max_episode_length $MAX_EPISODE_LENGTH --eval_frequency $EVAL_FREQUENCY --print_frequency $PRINT_FREQUENCY --save_frequency $SAVE_FREQUENCY --seed $SEED"

echo "Starting Gaussian DQL training on CUDA:1..."
tmux new-session -d -s train_gaussian_dql "python train.py --agent gaussian_dql --device cuda:1 --max_steps $MAX_TRAINING_STEPS --max_episode_length $MAX_EPISODE_LENGTH --eval_frequency $EVAL_FREQUENCY --print_frequency $PRINT_FREQUENCY --save_frequency $SAVE_FREQUENCY --seed $SEED"

echo ""
echo "All training sessions started!"
echo "============================================================"
echo "Monitor training progress with:"
echo "  tmux attach -t train_ppo_diffusion"
echo "  tmux attach -t train_gaussian_ppo"
echo "  tmux attach -t train_ql_diffusion"
echo "  tmux attach -t train_gaussian_dql"
echo ""
echo "List all sessions: tmux list-sessions"
echo "Kill all sessions: tmux kill-server"
echo ""

# Wait for all training to complete
echo "Waiting for all training sessions to complete..."
echo "This may take a while depending on your hardware..."
echo ""

# Function to check if all sessions are finished
wait_for_completion() {
    while true; do
        running_sessions=$(tmux list-sessions 2>/dev/null | grep -E "train_(ppo_diffusion|gaussian_ppo|ql_diffusion|gaussian_dql)" | wc -l)
        if [ "$running_sessions" -eq 0 ]; then
            break
        fi
        echo "Still running: $running_sessions training sessions..."
        sleep 30
    done
}

# Wait for completion
wait_for_completion

echo ""
echo "============================================================"
echo "TRAINING COMPLETED!"
echo "============================================================"

echo ""
echo "Step 2: Running performance analysis..."
echo "============================================================"
python analyze_performance.py --agents ppo_diffusion ql_diffusion gaussian_ppo gaussian_dql --user_counts $USER_COUNTS --episodes $ANALYSIS_EPISODES --max_steps $ANALYSIS_STEPS

echo ""
echo "Step 3: Running convergence plots..."
echo "============================================================"
python convergence_plots.py --agents ppo_diffusion ql_diffusion gaussian_ppo gaussian_dql --num_users 10 --episodes $CONVERGENCE_EPISODES --max_steps $CONVERGENCE_STEPS --save_dir convergence_plots

echo ""
echo "Step 4: Creating final comprehensive plots..."
echo "============================================================"
python create_final_plots.py

echo ""
echo "Step 5: Organizing output files..."
echo "============================================================"
# Move all generated files to final_plot directory
mv *.png *.csv final_plot/ 2>/dev/null || true
mv analysis_plots/ final_plot/ 2>/dev/null || true
mv convergence_plots/ final_plot/ 2>/dev/null || true
mv logs/ final_plot/ 2>/dev/null || true

echo ""
echo "============================================================"
echo "ANALYSIS PIPELINE COMPLETED!"
echo "============================================================"
echo "All results saved in: final_plot/"
echo ""
echo "Generated files:"
ls -la final_plot/
echo ""
echo "Training logs saved in: final_plot/logs/"
echo "Analysis complete! Check the final_plot/ directory for all results."