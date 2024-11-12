from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.spatial.distance import directed_hausdorff
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'src/results_objectives.csv'

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
    print(df.head())

    distancia_manhattan = pdist(df, metric='cityblock')
    matriz_distancia = squareform(distancia_manhattan) 

    modelo_clustering = AgglomerativeClustering(linkage='single', metric='manhattan', n_clusters=2)
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

def silouette_score(df:pd.DataFrame, cluster_models : list, filename = 'silouette'):
    scores = [silouette_score(df, model.fit_predict(df)) for model in cluster_models]
    plt.bar(silouette_score)
    
clustering(objs_path='src/results_objectives.csv')