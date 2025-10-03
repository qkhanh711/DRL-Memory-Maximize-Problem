import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import os

# Try to import seaborn, but don't require it
try:
    import seaborn as sns
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Set style
plt.style.use('default')

def create_comprehensive_analysis():
    """Create comprehensive analysis plots for the DRL Memory Maximize Problem"""
    
    # Read the data
    df = pd.read_csv('analysis_plots/comprehensive_analysis.csv')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Define colors for each agent
    agent_colors = {
        'gaussian_ppo': '#1f77b4',
        'gaussian_a2c': '#ff7f0e', 
        'gaussian_dql': '#2ca02c',
        'a2c_diffusion': '#d62728',
        'bc_diffusion': '#9467bd',
        'ql_diffusion': '#8c564b'
    }
    
    # 1. Reward Convergence (Top Left)
    ax1 = plt.subplot(3, 3, 1)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax1.plot(agent_data['Users'], agent_data['Final_Reward'], 
                marker='o', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(), 
                color=agent_colors.get(agent, 'black'))
    
    ax1.set_title('Reward vs Number of Users', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Users')
    ax1.set_ylabel('Final Reward')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Memory Usage (Top Middle)
    ax2 = plt.subplot(3, 3, 2)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax2.plot(agent_data['Users'], agent_data['Avg_Memory'], 
                marker='s', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(),
                color=agent_colors.get(agent, 'black'))
    
    ax2.set_title('Memory Usage vs Number of Users', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Users')
    ax2.set_ylabel('Average Memory Usage')
    ax2.grid(True, alpha=0.3)
    
    # 3. Latency (Top Right)
    ax3 = plt.subplot(3, 3, 3)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax3.plot(agent_data['Users'], agent_data['Avg_Latency'], 
                marker='^', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(),
                color=agent_colors.get(agent, 'black'))
    
    ax3.set_title('Latency vs Number of Users', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Number of Users')
    ax3.set_ylabel('Average Latency (seconds)')
    ax3.grid(True, alpha=0.3)
    
    # 4. QoS (Middle Left)
    ax4 = plt.subplot(3, 3, 4)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax4.plot(agent_data['Users'], agent_data['Avg_QoS'], 
                marker='d', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(),
                color=agent_colors.get(agent, 'black'))
    
    ax4.set_title('QoS vs Number of Users', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Number of Users')
    ax4.set_ylabel('Average QoS (BRISQUE score)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Denoise Steps (Middle Middle)
    ax5 = plt.subplot(3, 3, 5)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax5.plot(agent_data['Users'], agent_data['Avg_Denoise_Steps'], 
                marker='v', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(),
                color=agent_colors.get(agent, 'black'))
    
    ax5.set_title('Denoise Steps vs Number of Users', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Number of Users')
    ax5.set_ylabel('Average Denoise Steps')
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Heatmap (Middle Right)
    ax6 = plt.subplot(3, 3, 6)
    pivot_table = df.pivot(index='Agent', columns='Users', values='Final_Reward')
    
    if HAS_SEABORN:
        sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='YlOrRd', 
                    ax=ax6, cbar_kws={'label': 'Final Reward'})
    else:
        im = ax6.imshow(pivot_table.values, cmap='YlOrRd', aspect='auto')
        ax6.set_xticks(range(len(pivot_table.columns)))
        ax6.set_yticks(range(len(pivot_table.index)))
        ax6.set_xticklabels(pivot_table.columns)
        ax6.set_yticklabels(pivot_table.index)
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                text = ax6.text(j, i, f'{pivot_table.iloc[i, j]:.0f}',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax6, label='Final Reward')
    
    ax6.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Number of Users')
    ax6.set_ylabel('Agent')
    
    # 7. Memory Efficiency (Bottom Left)
    ax7 = plt.subplot(3, 3, 7)
    # Calculate memory efficiency (reward per memory unit)
    df['Memory_Efficiency'] = df['Final_Reward'] / df['Avg_Memory']
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax7.plot(agent_data['Users'], agent_data['Memory_Efficiency'], 
                marker='o', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(),
                color=agent_colors.get(agent, 'black'))
    
    ax7.set_title('Memory Efficiency vs Number of Users', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Number of Users')
    ax7.set_ylabel('Reward per Memory Unit')
    ax7.grid(True, alpha=0.3)
    
    # 8. Latency Efficiency (Bottom Middle)
    ax8 = plt.subplot(3, 3, 8)
    # Calculate latency efficiency (reward per latency unit)
    df['Latency_Efficiency'] = df['Final_Reward'] / (df['Avg_Latency'] + 1e-6)
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        ax8.plot(agent_data['Users'], agent_data['Latency_Efficiency'], 
                marker='s', linewidth=2, markersize=8, 
                label=agent.replace('_', ' ').title(),
                color=agent_colors.get(agent, 'black'))
    
    ax8.set_title('Latency Efficiency vs Number of Users', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Number of Users')
    ax8.set_ylabel('Reward per Latency Unit')
    ax8.grid(True, alpha=0.3)
    
    # 9. Summary Statistics (Bottom Right)
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary statistics
    summary_text = "PERFORMANCE SUMMARY\n\n"
    
    # Best overall performance
    best_overall = df.loc[df['Final_Reward'].idxmax()]
    summary_text += f"Best Overall: {best_overall['Agent'].replace('_', ' ').title()}\n"
    summary_text += f"Reward: {best_overall['Final_Reward']:.0f}\n"
    summary_text += f"Users: {best_overall['Users']}\n\n"
    
    # Most efficient memory
    best_memory = df.loc[df['Memory_Efficiency'].idxmax()]
    summary_text += f"Most Memory Efficient:\n{best_memory['Agent'].replace('_', ' ').title()}\n"
    summary_text += f"Efficiency: {best_memory['Memory_Efficiency']:.0f}\n\n"
    
    # Most efficient latency
    best_latency = df.loc[df['Latency_Efficiency'].idxmax()]
    summary_text += f"Most Latency Efficient:\n{best_latency['Agent'].replace('_', ' ').title()}\n"
    summary_text += f"Efficiency: {best_latency['Latency_Efficiency']:.0f}\n\n"
    
    # Average performance by user count
    summary_text += "Avg Performance by Users:\n"
    for users in sorted(df['Users'].unique()):
        avg_reward = df[df['Users'] == users]['Final_Reward'].mean()
        summary_text += f"{users} users: {avg_reward:.0f}\n"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive analysis plot saved as: comprehensive_analysis.png")

def create_convergence_plot():
    """Create detailed convergence plot"""
    # This would require the raw episode data, but we can create a simulated one
    # based on the final rewards
    df = pd.read_csv('analysis_plots/comprehensive_analysis.csv')
    
    plt.figure(figsize=(12, 8))
    
    # Simulate convergence curves based on final performance
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        final_reward = agent_data['Final_Reward'].mean()
        
        # Create simulated convergence curve
        episodes = np.arange(1, 16)
        # Simulate learning curve with some noise
        convergence = final_reward * (1 - np.exp(-episodes/5)) + np.random.normal(0, final_reward*0.1, len(episodes))
        convergence = np.maximum(convergence, 0)  # Ensure non-negative
        
        plt.plot(episodes, convergence, marker='o', linewidth=2, markersize=6, 
                label=agent.replace('_', ' ').title())
    
    plt.title('Reward Convergence Comparison\n(Simulated Learning Curves)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Convergence analysis plot saved as: convergence_analysis.png")

def create_detailed_comparison():
    """Create detailed comparison table and plots"""
    df = pd.read_csv('analysis_plots/comprehensive_analysis.csv')
    
    # Create detailed comparison table
    comparison_stats = []
    
    for agent in df['Agent'].unique():
        agent_data = df[df['Agent'] == agent]
        stats = {
            'Agent': agent.replace('_', ' ').title(),
            'Avg_Reward': agent_data['Final_Reward'].mean(),
            'Max_Reward': agent_data['Final_Reward'].max(),
            'Min_Reward': agent_data['Final_Reward'].min(),
            'Avg_Memory': agent_data['Avg_Memory'].mean(),
            'Avg_Latency': agent_data['Avg_Latency'].mean(),
            'Avg_QoS': agent_data['Avg_QoS'].mean(),
            'Avg_Denoise_Steps': agent_data['Avg_Denoise_Steps'].mean(),
            'Memory_Efficiency': (agent_data['Final_Reward'] / agent_data['Avg_Memory']).mean(),
            'Latency_Efficiency': (agent_data['Final_Reward'] / (agent_data['Avg_Latency'] + 1e-6)).mean()
        }
        comparison_stats.append(stats)
    
    comparison_df = pd.DataFrame(comparison_stats)
    comparison_df = comparison_df.sort_values('Avg_Reward', ascending=False)
    
    # Save detailed comparison
    comparison_df.to_csv('detailed_agent_comparison.csv', index=False)
    
    # Create ranking plot
    plt.figure(figsize=(12, 8))
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward ranking
    ax1 = axes[0, 0]
    bars1 = ax1.barh(comparison_df['Agent'], comparison_df['Avg_Reward'], 
                     color=plt.cm.viridis(np.linspace(0, 1, len(comparison_df))))
    ax1.set_title('Average Reward Ranking', fontweight='bold')
    ax1.set_xlabel('Average Reward')
    
    # Memory efficiency ranking
    ax2 = axes[0, 1]
    bars2 = ax2.barh(comparison_df['Agent'], comparison_df['Memory_Efficiency'], 
                     color=plt.cm.plasma(np.linspace(0, 1, len(comparison_df))))
    ax2.set_title('Memory Efficiency Ranking', fontweight='bold')
    ax2.set_xlabel('Reward per Memory Unit')
    
    # Latency efficiency ranking
    ax3 = axes[1, 0]
    bars3 = ax3.barh(comparison_df['Agent'], comparison_df['Latency_Efficiency'], 
                     color=plt.cm.inferno(np.linspace(0, 1, len(comparison_df))))
    ax3.set_title('Latency Efficiency Ranking', fontweight='bold')
    ax3.set_xlabel('Reward per Latency Unit')
    
    # QoS ranking (lower is better)
    ax4 = axes[1, 1]
    bars4 = ax4.barh(comparison_df['Agent'], comparison_df['Avg_QoS'], 
                     color=plt.cm.magma(np.linspace(0, 1, len(comparison_df))))
    ax4.set_title('QoS Ranking (Lower is Better)', fontweight='bold')
    ax4.set_xlabel('Average QoS (BRISQUE score)')
    
    plt.tight_layout()
    plt.savefig('agent_rankings.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Agent rankings plot saved as: agent_rankings.png")
    print("Detailed comparison table saved as: detailed_agent_comparison.csv")
    
    return comparison_df

def main():
    print("Creating comprehensive analysis plots...")
    
    # Create all plots
    create_comprehensive_analysis()
    create_convergence_plot()
    comparison_df = create_detailed_comparison()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("- comprehensive_analysis.png: Complete analysis dashboard")
    print("- convergence_analysis.png: Reward convergence comparison")
    print("- agent_rankings.png: Performance rankings")
    print("- detailed_agent_comparison.csv: Detailed statistics")
    
    print("\nTop 3 Agents by Average Reward:")
    for i, row in comparison_df.head(3).iterrows():
        print(f"{i+1}. {row['Agent']}: {row['Avg_Reward']:.0f}")

if __name__ == "__main__":
    main()