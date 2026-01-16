#!/bin/bash

set -euo pipefail

echo "============================================================"
echo "Run training and plot metrics for DRL models"
echo "============================================================"

# Default configuration for 4 models: DiffPPO, DiffQL, PPO, DQL
AGENTS="ppo_diffusion ql_diffusion gaussian_ppo gaussian_dql"
EPISODES=1000
MAX_STEPS=100000
NUM_USERS=10
SEED=42
QOS_TARGETS="25 30 35"
USER_COUNTS="8 10 12"

# Shortcuts:
#   bash run.sh train              -> Train all 4 models (convergence analysis)
#   bash run.sh analyze            -> Performance analysis across user counts
#   bash run.sh qos-sweep          -> QoS sweep analysis
#   bash run.sh full               -> Complete training and analysis pipeline
#   bash run.sh convergence        -> Custom convergence run
#   bash run.sh replot-analyze     -> Replot analysis only
#   bash run.sh replot-convergence -> Replot convergence only
#   bash run.sh dashboard          -> Dashboard view

CMD_ARGS=("$@")

if [ ${#CMD_ARGS[@]} -eq 0 ] || [ "${CMD_ARGS[0]}" = "help" ]; then
	echo "Usage:"
	echo ""
	echo "Quick Commands:"
	echo "  bash run.sh train              - Train 4 models (DiffPPO, DiffQL, PPO, DQL) with convergence plots"
	echo "  bash run.sh analyze            - Performance analysis across different user counts (8, 10, 12)"
	echo "  bash run.sh qos-sweep          - QoS target sweep analysis (25, 30, 35)"
	echo "  bash run.sh full               - Run complete pipeline (train + analyze + qos-sweep)"
	echo ""
	echo "Advanced Commands:"
	echo "  bash run.sh convergence [opts] - Custom convergence training"
	echo "  bash run.sh replot-analyze     - Replot performance analysis from saved data"
	echo "  bash run.sh replot-convergence - Replot convergence from saved data"
	echo "  bash run.sh dashboard          - Show training dashboard"
	echo ""
	echo "Options:"
	echo "  --agents <list>         - Space-separated agent names (default: $AGENTS)"
	echo "  --episodes <n>          - Number of episodes (default: $EPISODES)"
	echo "  --max_steps <n>         - Max steps per run (default: $MAX_STEPS)"
	echo "  --num_users <n>         - Number of users for convergence (default: $NUM_USERS)"
	echo "  --seed <n>              - Random seed (default: $SEED)"
	echo ""
	echo "Example:"
	echo "  bash run.sh train --episodes 500 --max_steps 50000"
	exit 0
fi

# Map shortcuts to full arguments
case "${CMD_ARGS[0]:-}" in
	train)
		echo "=========================================="
		echo "Training 4 models: DiffPPO, DiffQL, PPO, DQL"
		echo "=========================================="
		CMD_ARGS=("--run_convergence" "--agents" $AGENTS "--num_users" "$NUM_USERS" 
		          "--episodes" "$EPISODES" "--max_steps" "$MAX_STEPS" 
		          "--save_dir" "convergence_plots" "--seed" "$SEED" "${CMD_ARGS[@]:1}")
		;;
	analyze)
		echo "=========================================="
		echo "Performance analysis across user counts"
		echo "=========================================="
		CMD_ARGS=("--run_performance" "--agents" $AGENTS "--user_counts" $USER_COUNTS 
		          "--episodes" "$EPISODES" "--max_steps" "$MAX_STEPS" "--seed" "$SEED" "${CMD_ARGS[@]:1}")
		;;
	qos-sweep)
		echo "=========================================="
		echo "QoS target sweep analysis"
		echo "=========================================="
		mkdir -p final_plot
		python scripts/analyze_performance.py --agents $AGENTS \
		  --qos_requireds $QOS_TARGETS --user_counts $USER_COUNTS \
		  --episodes 100 --max_steps 10000 --device auto --seed $SEED "${CMD_ARGS[@]:1}"
		move_dir "analysis_plots"
		echo "QoS sweep complete!"
		exit 0
		;;
	full)
		echo "=========================================="
		echo "Running complete pipeline"
		echo "=========================================="
		# 1. Training with convergence
		echo ""
		echo "[1/3] Training models..."
		python convergence_analyze.py --run_convergence --agents $AGENTS \
		  --num_users $NUM_USERS --episodes $EPISODES --max_steps $MAX_STEPS \
		  --save_dir convergence_plots --seed $SEED
		
		# 2. Performance analysis
		echo ""
		echo "[2/3] Performance analysis..."
		python convergence_analyze.py --run_performance --agents $AGENTS \
		  --user_counts $USER_COUNTS --episodes $EPISODES --max_steps $MAX_STEPS --seed $SEED
		
		# 3. QoS sweep
		echo ""
		echo "[3/3] QoS sweep analysis..."
		python scripts/analyze_performance.py --agents $AGENTS \
		  --qos_requireds $QOS_TARGETS --user_counts $USER_COUNTS \
		  --episodes 100 --max_steps 10000 --device auto --seed $SEED
		
		# Organize results
		mkdir -p final_plot
		move_dir "convergence_plots"
		move_dir "analysis_plots"
		echo ""
		echo "Complete pipeline finished! Results in final_plot/"
		exit 0
		;;
	convergence)
		CMD_ARGS=("--run_convergence" "${CMD_ARGS[@]:1}")
		;;
	replot-analyze)
		CMD_ARGS=("--replot" "--run_performance" "${CMD_ARGS[@]:1}")
		;;
	replot-convergence)
		CMD_ARGS=("--replot" "--run_convergence" "${CMD_ARGS[@]:1}")
		;;
	dashboard)
		CMD_ARGS=("--run_dashboard" "${CMD_ARGS[@]:1}")
		;;
esac

# Default values (match convergence_analyze.py defaults)
SAVE_DIR="convergence_plots"

# Parse --save_dir from args (so we can move it later)
ARGS=("${CMD_ARGS[@]}")
for (( i=0; i<${#ARGS[@]}; i++ )); do
	if [ "${ARGS[$i]}" = "--save_dir" ]; then
		if [ $((i+1)) -lt ${#ARGS[@]} ]; then
			SAVE_DIR="${ARGS[$((i+1))]}"
		fi
	fi
done

mkdir -p final_plot

echo "Using save_dir: ${SAVE_DIR}"
echo "Running: python convergence_analyze.py ${CMD_ARGS[*]}"
python convergence_analyze.py "${CMD_ARGS[@]}"

echo ""
echo "Moving generated folders into final_plot..."

# Helper to move a directory into final_plot safely
move_dir() {
	local src_dir="$1"
	if [ -d "$src_dir" ]; then
		local dst_dir="final_plot/$(basename "$src_dir")"
		if [ -d "$dst_dir" ]; then
			rm -rf "$dst_dir"
		fi
		mv "$src_dir" final_plot/
		echo "Moved: $src_dir -> final_plot/"
	fi
}

# Move known plot folders
move_dir "$SAVE_DIR"
move_dir "analysis_plots"

echo "Done. Contents of final_plot/:"
ls -la final_plot/ || true

echo "All folders have been organized into final_plot."