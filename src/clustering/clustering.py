from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
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
        data_filename = data_path.split('_')[-1].replace('.csv', '')
        self.out_path = f"{out_path}/{self.local_time}-{data_filename}-{clustering_alg}"
        self.data_df = self.normalize_data(pd.read_csv(data_path))
        self.metric = metric
        self.clustering_alg =clustering_alg

        ## single, ward, average, complete
        self.linkage = 'single'
    
        os.makedirs(self.out_path, exist_ok=True)

    def cluster_optimal(self):
        n_clusters = self.optimal_cluster_size

        if self.clustering_alg == 'aglomerative':
                model = AgglomerativeClustering(linkage=self.linkage, 
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
                model = AgglomerativeClustering(linkage=self.linkage, 
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
        if 'log_name' in df.columns:
            df.drop('log_name', inplace=True, axis=1)
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

        plt.title('Cluster Sizes ', fontsize=14, fontweight='bold')
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


class HittingSearchClustering(Clustering):

    def hitting_search(self, max_clusters):
        methods = {
            'kmeans': None,
            'aglomerative_single': 'single',
            'aglomerative_ward': 'ward',
            'aglomerative_average': 'average',
            'aglomerative_complete': 'complete'
        }

        results = {}
        
        # Iterar por cada algoritmo y realizar el clustering
        for method, linkage in methods.items():
            print(f"\n### Running clustering with method: {method} ###\n")
            self.clustering_alg = 'kmeans' if method == 'kmeans' else 'aglomerative'
            self.linkage = linkage
            
            # Calcular número de clústeres óptimo y realizar clustering
            self.cluster_test(max_clusters)
            n_clusters = self.optimal_cluster_size

            if self.clustering_alg == 'kmeans':
                model = KMeans(n_clusters=n_clusters, 
                               init="k-means++", 
                               n_init=10,
                               max_iter=280, 
                               random_state=42)
            else:
                model = AgglomerativeClustering(linkage=self.linkage, 
                                                metric=self.metric, 
                                                n_clusters=n_clusters)
                
            predict = model.fit_predict(self.data_df)
            
            # Guardar resultados
            results[method] = {
                'clusters': n_clusters,
                'labels': predict,
                'silhouette': silhouette_score(self.data_df, predict),
            }
            print(f"Optimal clusters for {method}: {n_clusters}")
            print(f"Silhouette score: {results[method]['silhouette']:.3f}\n")

        # Comparar resultados
        self.compare_results(results)
    
    def compare_results(self, results):
        print("\n### Comparing Results ###\n")

        all_methods = list(results.keys())
        comparisons = []

        for i in range(len(all_methods)):
            for j in range(i + 1, len(all_methods)):
                method1, method2 = all_methods[i], all_methods[j]
                labels1, labels2 = results[method1]['labels'], results[method2]['labels']
                
                # Alinear etiquetas de clústeres
                aligned_labels2 = self.align_labels(labels1, labels2)
                
                # Comparar etiquetas alineadas
                matches = np.sum(labels1 == aligned_labels2) / len(labels1)
                comparisons.append((method1, method2, matches))
                print(f"Comparison {method1} vs {method2}: {matches * 100:.2f}% of points match.")
        
        # Visualizar diferencias en PCA
        self.visualize_comparisons(results)
    
    def align_labels(self, labels1, labels2):
        """
        Alinear etiquetas de clústeres entre dos conjuntos de etiquetas para maximizar coincidencias.
        """
        # Crear matriz de contingencia
        contingency_matrix = confusion_matrix(labels1, labels2)
        
        # Resolver asignación óptima usando el algoritmo húngaro
        row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
        
        # Crear un mapa de etiquetas alineadas
        label_mapping = {old: new for old, new in zip(col_ind, row_ind)}
        
        # Asignar las etiquetas alineadas
        aligned_labels2 = np.array([label_mapping[label] for label in labels2])
        return aligned_labels2
    
    def visualize_comparisons(self, results):
        """
        Generar visualizaciones de comparación entre diferentes algoritmos.
        Incluye los gráficos de clusters en PCA para cada algoritmo y un gráfico de barras con comparaciones.
        """
        all_methods = list(results.keys())
        comparisons = []

        # Configurar el gráfico con subplots
        fig, axs = plt.subplots(2, 3, figsize=(24, 16))  # 2 filas x 3 columnas
        fig.suptitle("Comparison of Clustering Methods", fontsize=18, fontweight="bold")

        # Subplots 1-5: PCA de cada algoritmo
        for idx, method in enumerate(all_methods):
            data = self.data_df.copy()
            data['Cluster'] = results[method]['labels']

            # Reducir a 2 dimensiones con PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data.drop(columns=['Cluster']))
            data['PCA1'], data['PCA2'] = pca_result[:, 0], pca_result[:, 1]

            # Graficar
            sns.scatterplot(ax=axs[idx // 3, idx % 3],
                            x=data['PCA1'], y=data['PCA2'],
                            hue=data['Cluster'], palette="hsv", s=60)
            axs[idx // 3, idx % 3].set_title(f"Clusters with PCA: {method}")
            axs[idx // 3, idx % 3].set_xlabel("PCA1")
            axs[idx // 3, idx % 3].set_ylabel("PCA2")
            axs[idx // 3, idx % 3].legend(title='Cluster', loc='best')

        # Comparar pares de algoritmos
        for i in range(len(all_methods)):
            for j in range(i + 1, len(all_methods)):
                method1, method2 = all_methods[i], all_methods[j]
                labels1, labels2 = results[method1]['labels'], results[method2]['labels']
                
                # Alinear etiquetas
                aligned_labels2 = self.align_labels(labels1, labels2)
                
                # Calcular porcentaje de coincidencia
                matches = np.sum(labels1 == aligned_labels2) / len(labels1)
                comparisons.append((f"{method1} vs {method2}", matches))

        # Subplot 6: Comparaciones ordenadas
        comparisons.sort(key=lambda x: x[1], reverse=True)
        labels, scores = zip(*comparisons)

        axs[1, 2].barh(labels, scores, color='skyblue')
        axs[1, 2].set_title("Pairwise Cluster Comparison")
        axs[1, 2].set_xlabel("Match Percentage")
        axs[1, 2].invert_yaxis()  # Para mostrar el más alto primero

        # Ajustar diseño
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Deja espacio para el título principal
        plt.savefig(f"{self.out_path}/comparisons_full.png")
        plt.close()

if __name__ == "__main__":

    closed =    'src/results_objectives_closed.csv'
    financial = 'src/results_objectives_financial.csv'
    open =      'src/results_objectives_open.csv'
    incidents = 'src/results_objectives_incidents.csv'

    completo = 'src/csv_completo.csv'
    closed_data = 'data-closed.csv'
    open_data = 'data-open.csv'

    clust = HittingSearchClustering(metric='euclidean',
                                    clustering_alg='aglomerative',
                                    data_path=open_data ,
                                    out_path='src/tests/out/hitting_search')

    clust.hitting_search(max_clusters=10)

    
"""
clust = Clustering(metric='euclidean',
                    clustering_alg='kmeans',
                    data_path=completo,
                    out_path='src/tests/out/clustering')

clust.cluster_test(max_clusters=10)"""
