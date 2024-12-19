import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def visualize_process_mining_pca(csv_path):
    """
    Visualize process mining metrics using PCA
    
    Parameters:
    csv_path (str): Path to the CSV file containing metrics
    
    Returns:
    PCA transformed data and matplotlib figure
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    metrics = ['precision', 'fitness', 'simplicity', 'generalization']
    X = df[metrics]
    
    #scaler = StandardScaler()
    #X_scaled = scaler.fit_transform(X)
    
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(10, 6))
    
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=X_pca[:, 0],  # Color based on first PC
                          cmap='viridis', 
                          alpha=0.7)
    
    plt.colorbar(scatter, label='First Principal Component')
    
    plt.title('PCA Visualization of Process Mining Metrics', fontsize=14)
    plt.xlabel('First Principal Component\n(Precision & Fitness)', fontsize=10)
    plt.ylabel('Second Principal Component\n(Simplicity & Generalization)', fontsize=10)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    
    explained_variance = pca.explained_variance_ratio_
    plt.annotate(
        f'Explained Variance:\nPC1: {explained_variance[0]*100:.2f}%\nPC2: {explained_variance[1]*100:.2f}%',
        xy=(0.05, 0.95), 
        xycoords='axes fraction',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    print("PCA Components:")
    pca_components = pd.DataFrame(
        pca.components_[:2].T, 
        columns=['First PC', 'Second PC'], 
        index=metrics
    )
    print(pca_components)
    
    return X_pca, plt

def main(path):
    csv_path = path
    X_pca, plt = visualize_process_mining_pca(csv_path)
    plt.savefig('process_mining_pca.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    csv_data = 'src/results_qual_objectives_financial.csv'
    main(csv_data)