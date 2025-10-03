#!/bin/bash

set -euo pipefail

echo "============================================================"
echo "Run convergence_analyze.py with passed or shortcut commands"
echo "============================================================"

# Shortcuts:
#   bash run.sh analyze [extra args]            -> --run_performance
#   bash run.sh convergence [extra args]        -> --run_convergence
#   bash run.sh replot-analyze [extra args]     -> --replot --run_performance
#   bash run.sh replot-convergence [extra args] -> --replot --run_convergence
#   bash run.sh dashboard [extra args]          -> --run_dashboard

CMD_ARGS=("$@")

if [ ${#CMD_ARGS[@]} -eq 0 ] || [ "${CMD_ARGS[0]}" = "help" ]; then
	echo "Usage:"
	echo "  bash run.sh analyze [--user_counts 8 10 12] [--episodes 50] [--max_steps 1000]"
	echo "  bash run.sh convergence [--agents ...] [--num_users 10] [--episodes 100] [--max_steps 2000] [--save_dir convergence_plots]"
	echo "  bash run.sh replot-analyze"
	echo "  bash run.sh replot-convergence [--save_dir convergence_plots]"
	echo "  bash run.sh dashboard"
	echo "  bash run.sh [direct args to convergence_analyze.py]"
fi

# Map shortcuts to full arguments
case "${CMD_ARGS[0]:-}" in
	analyze)
		CMD_ARGS=("--run_performance" "${CMD_ARGS[@]:1}")
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