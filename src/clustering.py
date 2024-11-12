from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import directed_hausdorff
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import scipy.cluster.hierarchy as shc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

def minimum_pairwise_distance(cluster_a, cluster_b):
    # Calcula la distancia mínima entre puntos de dos clusters
    min_dist = np.min([np.sum(np.abs(a - b)) for a in cluster_a for b in cluster_b])
    return min_dist

def averaged_hausdorff_distance(cluster_a, cluster_b):
    # Cluster_a y cluster_b deben ser subconjuntos del DataFrame original
    ahd = (directed_hausdorff(cluster_a, cluster_b)[0] + directed_hausdorff(cluster_b, cluster_a)[0]) / 2
    return ahd

def normalize_data(df: pd.DataFrame):
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return df_normalized

def clustering(objs_path: str):
        
    df = pd.read_csv(objs_path)
    df.drop_duplicates(inplace=True)
    df['fitness'] = df['fitness'].apply(abs)
    df['precission'] = df['precission'].apply(abs)
    
    df = normalize_data(df)

    modelo_clustering = AgglomerativeClustering(linkage='single', metric='manhattan', n_clusters=4)
    etiquetas_clusters = modelo_clustering.fit_predict(df)

    df['Cluster'] = etiquetas_clusters

    # Calcular AHD entre todos los pares de clusters
    clusters_unicos = df['Cluster'].unique()
    ahd_matriz = pd.DataFrame(index=clusters_unicos, columns=clusters_unicos)

    for i in clusters_unicos:
        for j in clusters_unicos:
            if i != j:
                ahd_matriz.loc[i, j] = averaged_hausdorff_distance(df[df['Cluster'] == i], df[df['Cluster'] == j])

    # Calcular MPD entre todos los pares de clusters
    mpd_matriz = pd.DataFrame(index=clusters_unicos, columns=clusters_unicos)

    for i in clusters_unicos:
        for j in clusters_unicos:
            if i != j:
                mpd_matriz.loc[i, j] = minimum_pairwise_distance(df[df['Cluster'] == i].values, df[df['Cluster'] == j].values)

    plot_PCA(df)
    plot_dendogram(df)
    print(df.head())

    

def plot_PCA(df: pd.DataFrame, filename = 'cluster'):
    # Reducir a 2 dimensiones con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop(columns=['Cluster']))

    # Agregar los resultados de PCA al DataFrame
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]

    # Visualizar los clusters en el espacio reducido
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['PCA1'], y=df['PCA2'], hue=df['Cluster'], palette='bright', s=60)
    plt.title("Clusters visualizados en el espacio PCA")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.legend(title='Cluster')
    plt.savefig(filename)

def plot_dendogram(df: pd.DataFrame, filename = 'dendogram'):
    plt.figure(figsize=(10, 6))
    plt.title("Dendograma")
    shc.dendrogram(shc.linkage(df, method='ward'))
    plt.savefig(filename)


    
#clustering(objs_path='src/results_objectives.csv')


class Clustering():
    
    def __init__(self, max_clusters, metric, data_path, out_path):
        self.max_clusters = max_clusters
        self.data_path = data_path
        self.local_time = time.strftime("[%Y_%m_%d - %H:%M:%S]", time.localtime())
        self.out_path = f'{out_path}/{self.local_time}'
        self.data_df = self.normalize_data(pd.read_csv(data_path))
        self.metric = metric
    
        os.makedirs(self.out_path, exist_ok=True)
        
    def cluster(self):
        metrics = list()
        for n_clusters in range(2, self.max_clusters+1):
            model_ac = AgglomerativeClustering(linkage='single', metric=self.metric, n_clusters = n_clusters)
            predict = model_ac.fit_predict(self.data_df)
            labels = pd.DataFrame(predict, columns=['Labels'], index=self.data_df.index)
            self.plot_silhouettes_and_objects(labels, predict, filename=f'silhouette_objects_{n_clusters}')
            metrics.append(silhouette_score(self.data_df, labels['Labels']))
        self.silhouette_result(metrics)
        
    def normalize_data(self, df: pd.DataFrame):
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_normalized
    
    def silhouette_result(self, metrics_score:list, filename='silouette_result'):
        plt.figure(figsize=(25, 8))
        sns.set_style("darkgrid")
        plt.title(f'Silhouette score for different number of clusters', fontsize=14, fontweight='bold')
        plt.xlabel('Clusters', fontsize=14, fontweight='bold')
        plt.ylabel('Silhouette', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(list(range(2, self.max_clusters + 1)), metrics_score, marker='o')
        plt.savefig(f'{self.out_path}/{filename}.png')
        plt.close()

    def plot_silhouettes_and_objects(self, labels, predict, filename='silhouette_objects'):
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
        sns.scatterplot(x=data['PCA1'], y=data['PCA2'], hue=data['Cluster'], palette='bright', s=60)
        plt.title("Clusters in PCA space", fontsize=14, fontweight='bold')
        plt.xlabel("PCA1", fontsize=14, fontweight='bold')
        plt.ylabel("PCA2", fontsize=14, fontweight='bold')
        plt.legend(title='Cluster')
        plt.savefig(filename)

        ## Guardar
        plt.tight_layout()
        plt.savefig(f'{self.out_path}/{filename}.png')
        plt.close()

clust = Clustering(max_clusters=40,
                   metric='euclidean',
                   data_path='src/results_objectives.csv',
                   out_path='src/tests/out/clustering')

clust.cluster()