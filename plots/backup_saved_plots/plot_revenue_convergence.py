import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d

def get_algorithm_display_name(alg):
    """Convert algorithm names to display names for plots and legends"""
    mapping = {
        'ppo_diffusion': 'GSRM-DiPO',
        'ql_diffusion': 'GSRM-DiQL',
        'gaussian_ppo': 'GSRM-PPO',
        'gaussian_dql': 'GSRM-DQL',
    }
    return mapping.get(alg, alg.replace('_', ' ').title())

def smooth_values(values, window=200):
    return uniform_filter1d(values, size=window)

def plotting_revenue(training_data, qos=[30.0], users=[8, 10, 12], is_out=False, colors=None, window=1000):
    plt.figure(figsize=(5, 4))
    
    # Set style
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.7
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    for i in range(len(qos)):
        for j in range(len(users)):
            for alg_name in training_data.keys():
                label = f"{get_algorithm_display_name(alg_name)}"
                
                # Check availability
                if str(qos[i]) not in training_data[alg_name] or \
                   str(users[j]) not in training_data[alg_name][str(qos[i])]:
                    print(f"Skipping {alg_name} (data not found for QoS={qos[i]}, Users={users[j]})")
                    continue
                    
                revenues = training_data[alg_name][str(qos[i])][str(users[j])]
                
                if not revenues:
                    print(f"Skipping {alg_name} (empty data for QoS={qos[i]}, Users={users[j]})")
                    continue

                # Specific adjustment for ppo_diffusion if needed (from original notebook logic)
                # if alg_name == "ppo_diffusion":
                #     revenues = [r + 10 for r in revenues] 

                color = None
                if colors and alg_name in colors:
                    color = colors[alg_name][str(qos[i])][str(users[j])]
                
                plt.plot(smooth_values(revenues, window), 
                         label=label, 
                         color=color, 
                         alpha=0.8)

    plt.xlabel('Episode')
    plt.ylabel('Revenue')
    plt.xlim(0, 10000)
    
    if is_out:
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    else:
        plt.legend(fontsize=10, loc='lower right') # Changed to lower right for Revenue typically increasing
    
    plt.grid(True)
    plt.tight_layout()
    output_filename = 'revenue_convergence.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Saved plot to {output_filename}")

def main():
    seed = 25
    logs_dir = Path(f"/home/n2tp/projects/Khanh_stuff/DRL-Memory-Maximize-Problem/logs/{seed}")
    qos_required = [25.0, 30.0, 35.5]
    num_users = [8, 10, 12]
    
    # Collect data
    training_data = {}
    
    if not logs_dir.exists():
        print(f"Error: Directory {logs_dir} does not exist.")
        return

    for folder in sorted(logs_dir.iterdir()):
        if not folder.is_dir():
            continue
            
        print(f"Processing algorithm: {folder.name}")
        training_data[folder.name] = {}
        
        for qos in qos_required:
            training_data[folder.name][str(qos)] = {}
            for n_users in num_users:
                metrics_file = folder / f"convergence_metrics_qos_{qos}_users_{n_users}.json"
                
                if metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            data = json.load(f)
                            
                        metrics = []
                        if "episode_metrics" in data:
                            for ep in data["episode_metrics"]:
                                if "final_info" in ep and "total_revenue" in ep["final_info"]:
                                    metrics.append(ep["final_info"]["total_revenue"])
                        
                        if metrics:
                            training_data[folder.name][str(qos)][str(n_users)] = metrics
                            print(f"  Loaded QoS={qos}, Users={n_users}: {len(metrics)} episodes")
                        else:
                            print(f"  Warning: No revenue data found in {metrics_file.name}")
                            
                    except Exception as e:
                        print(f"  Error reading {metrics_file.name}: {e}")
                else:
                    # print(f"  File not found: {metrics_file.name}")
                    pass

    # Colors definition (copied from notebook)
    colors = {
         'ql_diffusion': {
            '25.0': {'8': '#B8860B', '10': '#FFD700', '12': 'tab:orange'},
            '30.0': {'8': '#DAA520', '10': '#F0E68C', '12': '#FFFACD'},
            '35.0': {'8': '#BDB76B', '10': '#EEE8AA', '12': '#FAFAD2'},
        },
        'ppo_diffusion': {
            '25.0': {'8': '#8B0000', '10': '#DC143C', '12': 'tab:red'},
            '30.0': {'8': '#B22222', '10': '#FF0000', '12': '#FFA07A'},
            '35.0': {'8': '#CD5C5C', '10': '#FF6347', '12': '#F08080'},
        },
        'gaussian_ppo': {
            '25.0': {'8': '#00008B', '10': '#4169E1', '12': 'tab:blue'},
            '30.0': {'8': '#0000CD', '10': '#1E90FF', '12': '#87CEFA'},
            '35.0': {'8': '#4682B4', '10': '#00BFFF', '12': '#B0E0E6'},
        },
        'gaussian_dql': {
            '25.0': {'8': '#006400', '10': '#32CD32', '12': 'tab:green'},
            '30.0': {'8': '#228B22', '10': '#00FF00', '12': '#98FB98'},
            '35.0': {'8': '#2E8B57', '10': '#00FA9A', '12': '#AFEEEE'},
        },
    }

    # Plot
    # Adjust parameters here to match the specific plot desired (e.g. QoS=25.0, Users=12)
    # The user wanted "Cell 21" behavior which used QoS=25.0, Users=12
    plotting_revenue(training_data, qos=[25.0], users=[12], is_out=False, colors=colors)

if __name__ == "__main__":
    main()
