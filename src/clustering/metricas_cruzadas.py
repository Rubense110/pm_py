import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_process_mining_metrics(csv_path):
    """
    Visualize process mining quality metrics in a 2D scatter plot.
    
    Parameters:
    csv_path (str): Path to the CSV file containing metrics
    
    Returns:
    matplotlib figure with two subplots
    """
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(
        data=df, 
        x='precision', 
        y='fitness', 
        ax=ax1,
        color='blue',
        alpha=0.7
    )
    ax1.set_title('Precision vs Fitness')
    ax1.set_xlabel('Precision')
    ax1.set_ylabel('Fitness')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    ax1.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)
    
    sns.scatterplot(
        data=df, 
        x='simplicity', 
        y='generalization', 
        ax=ax2,
        color='green',
        alpha=0.7
    )
    ax2.set_title('Simplicity vs Generalization')
    ax2.set_xlabel('Simplicity')
    ax2.set_ylabel('Generalization')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig.suptitle('Process Mining Quality Metrics', fontsize=16, y=1.05)
    
    return fig

def main(path):
    csv_path = path
    fig = visualize_process_mining_metrics(csv_path)
    fig.savefig('process_mining_metrics_visualization.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    csv_data = 'src/results_qual_objectives_financial.csv'
    main(csv_data)

