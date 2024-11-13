from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import directed_hausdorff
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from yellowbrick.cluster import KElbowVisualizer
import scipy.cluster.hierarchy as shc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os    

class Clustering():
    
    def __init__(self, metric, clustering_alg, data_path, out_path):
        self.data_path = data_path
        self.local_time = time.strftime("[%Y_%m_%d - %H:%M:%S]", time.localtime())
        data_filename = data_path.split('_')[-1].replace('csv', '')
        self.out_path = f"{out_path}/{self.local_time}-{data_filename}"
        self.data_df = self.normalize_data(pd.read_csv(data_path))
        self.metric = metric
        self.clustering_alg =clustering_alg
    
        os.makedirs(self.out_path, exist_ok=True)

    def cluster_optimal(self):
        n_clusters = self.optimal_cluster_size

        if self.clustering_alg == 'aglomerative':
                model = AgglomerativeClustering(linkage='single', 
                                                metric=self.metric, 
                                                n_clusters = n_clusters)
                predict = model.fit_predict(self.data_df)
                
        elif self.clustering_alg == 'kmeans':
                model = KMeans(n_clusters=n_clusters, 
                               init="k-means++", 
                               n_init=10,
                               max_iter=280, 
                               random_state=42)
                predict = model.fit_predict(self.data_df)
        
        labeled_data = self.data_df.copy()
        labeled_data['Cluster'] = predict

        canonical_representatives = self.get_canonical(labeled_data)

        for cluster_num in range(n_clusters):
            cluster_elements = labeled_data[labeled_data['Cluster'] == cluster_num]
            print(f"Clúster {cluster_num}:")
            print(cluster_elements)
            print("\n" + "="*40 + "\n")

        print("\n" + "#"*40)
        print("Representantes canónicos de cada clúster:")
        print("#"*40 + "\n")
        for cluster_id in sorted(canonical_representatives.keys()):  # Ordenar las claves antes de imprimir
            representative = canonical_representatives[cluster_id]
            print(f"Clúster {cluster_id} - Representante:\n{representative}")
            print("\n" + "="*40 + "\n")

        
    def cluster_test(self, max_clusters):
        metrics = list()
        distortions = list()

        self.elbow_visualization(max_clusters)
        for n_clusters in range(2, max_clusters+1):

            if self.clustering_alg == 'aglomerative':
                model = AgglomerativeClustering(linkage='single', 
                                                metric=self.metric, 
                                                n_clusters = n_clusters)
                predict = model.fit_predict(self.data_df)
                
            elif self.clustering_alg == 'kmeans':
                model = KMeans(n_clusters=n_clusters, 
                               init="k-means++", 
                               n_init=10,
                               max_iter=280, 
                               random_state=42)
                predict = model.fit_predict(self.data_df)
                distortion = model.inertia_
                distortions.append(distortion)
            
            labels = pd.DataFrame(predict, columns=['Labels'], index=self.data_df.index)
            self.plot_silhouettes_and_objects(labels, predict, n_clusters, filename=f'silhouette_objects_{n_clusters}')
            metrics.append(silhouette_score(self.data_df, labels['Labels']))

        self.silhouette_result(metrics, max_clusters)
        if self.clustering_alg == 'kmeans':
            self.kmeans_distortion(distortions, max_clusters)

        self.cluster_optimal()

    def get_canonical(self, labeled_data):
        representatives = {}
        clusters = labeled_data['Cluster'].unique()

        for cluster_id in clusters:
            cluster_data = labeled_data[labeled_data['Cluster'] == cluster_id]
            representative = cluster_data.sample(n=1).iloc[0]
            representatives[cluster_id] = representative

        return representatives

    def elbow_visualization(self, max_clusters, filename = 'elbow'):
        if self.clustering_alg == 'aglomerative':
            Elbow_M = KElbowVisualizer(AgglomerativeClustering(), k=max_clusters)
        elif self.clustering_alg == 'kmeans':
            Elbow_M = KElbowVisualizer(KMeans(), k=max_clusters)
        Elbow_M.fit(self.data_df)
        self.optimal_cluster_size = Elbow_M.elbow_value_
        Elbow_M.show(f'{self.out_path}/{filename}.png')
        
    def normalize_data(self, df: pd.DataFrame):
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_normalized
    
    def silhouette_result(self, metrics_score:list, max_clusters,filename='silouette_result'):
        plt.figure(figsize=(25, 8))
        sns.set_style("darkgrid")
        plt.title(f'Silhouette score for different number of clusters', fontsize=14, fontweight='bold')
        plt.xlabel('Clusters', fontsize=14, fontweight='bold')
        plt.ylabel('Silhouette', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(list(range(2, max_clusters + 1)), metrics_score, marker='o')
        plt.savefig(f'{self.out_path}/{filename}.png')
        plt.close()

    def kmeans_distortion(self, kmeans_distortions, max_clusters, filename='distortion'):
        plt.figure(figsize=(25, 8))
        sns.set_style("darkgrid")
        plt.title('Distortion values for different number of clusters (for Kmeans)', fontsize=14, fontweight='bold')
        plt.xlabel('Clusters', fontsize=14, fontweight='bold')
        plt.ylabel('SSE',fontsize=14, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(list(range(2, max_clusters + 1)), kmeans_distortions, marker='o')
        plt.savefig(f'{self.out_path}/{filename}.png')
        plt.close()

    def plot_silhouettes_and_objects(self, labels, predict, clusters, filename='silhouette_objects'):
        plt.figure(figsize=(25, 10))
        
        ## Subplot 1: Tamaño de clústeres
        plt.subplot(1, 3, 1)
        sns.set_style("darkgrid")
        cluster_group = labels.groupby('Labels').size()

        plt.title('Cluster Sizes for Different Models', fontsize=14, fontweight='bold')
        sns.barplot(x=cluster_group.values, y=list(map(str, cluster_group.index)))
        plt.xlabel('Cluster', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Objects', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        ## Subplot 2: Gráfico de Silueta
        plt.subplot(1, 3, 2)
        cluster_labels = np.unique(predict)
        silhouette_vals = silhouette_samples(self.data_df, predict, metric=self.metric)
        y_ax_lower, y_ax_upper = 0, 0
        yticks = []

        for i, с in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[predict == с]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            plt.barh(range(y_ax_lower, y_ax_upper),
                   c_silhouette_vals,
                   height = 1,
                   edgecolor='none')

            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals) 
        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--")
        sns.set_style("darkgrid")
        plt.title(f'Silhouette plot for different clusters', fontsize=14, fontweight='bold')   
        val = float(''.join([i for i in str(silhouette_avg)][0:5]))
        plt.xlabel(f'Silhouette = {val}', fontsize=14, fontweight='bold')
        plt.ylabel('Clusters', fontsize=14, fontweight='bold')
        plt.yticks(yticks, cluster_labels, fontsize=14)

        ## Subplot 3 : PCA
        data = self.data_df.copy()
        data['Cluster'] = predict
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(data.drop(columns=['Cluster']))

        data['PCA1'] = X_pca[:, 0]
        data['PCA2'] = X_pca[:, 1]

        plt.subplot(1, 3, 3)
        sns.scatterplot(x=data['PCA1'], y=data['PCA2'], hue=data['Cluster'], palette=sns.color_palette("hls", clusters), s=60)
        plt.title("Clusters in PCA space", fontsize=14, fontweight='bold')
        plt.xlabel("PCA1", fontsize=14, fontweight='bold')
        plt.ylabel("PCA2", fontsize=14, fontweight='bold')
        plt.legend(title='Cluster')

        ## Guardar
        plt.tight_layout()
        plt.savefig(f'{self.out_path}/{filename}.png')
        plt.close()

if __name__ == "__main__":

    closed = 'src/results_objectives_closed.csv'
    financial = 'src/results_objectives_financial.csv'
    open = 'src/results_objectives_open.csv'
    incidents = 'src/results_objectives_incidents.csv'

    clust = Clustering(metric='manhattan',
                    clustering_alg='kmeans',
                    data_path=incidents,
                    out_path='src/tests/out/clustering')

    clust.cluster_test(max_clusters=10)