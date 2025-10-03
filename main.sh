EPISODES=1000
MAX_STEPS=100000
SEED=42

# bash run.sh analyze --agents gaussian_ppo ql_diffusion gaussian_dql ppo_diffusion --user_counts 8 10 12 --episodes $EPISODES --max_steps $MAX_STEPS --seed $SEED
bash run.sh convergence --agents gaussian_ppo ql_diffusion gaussian_dql ppo_diffusion --num_users 10 --episodes $EPISODES --max_steps $MAX_STEPS --save_dir convergence_plots --seed $SEED
# bash run.sh replot-analyze
# bash run.sh replot-convergence --save_dir convergence_plots
# bash run.sh dashboard
# bash run.sh --run_performance --agents ql_diffusion  ppo_diffusion gaussian_dql gaussian_ppo --run_convergence --episodes $EPISODES --max_steps $MAX_STEPS
python scripts/analyze_performance.py --agents gaussian_ppo gaussian_dql ql_diffusion ppo_diffusion \
  --qos_requireds 25 30 35 \
  --episodes 50 --max_steps 1000 --device auto