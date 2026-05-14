from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

def get_silhouette(data, labels):
    """
    Calculate the Silhouette Score.
    """
    if len(set(labels)) > 1:
        return silhouette_score(data, labels)
    return -1

def calculate_davies_bouldin(data, labels):
    """
    Calculate the Davies-Bouldin Index.
    """
    if len(set(labels)) > 1:
        return davies_bouldin_score(data, labels)
    return -1

def calculate_calinski_harabasz(data, labels):
    """
    Calculate the Calinski-Harabasz Index.
    """
    if len(set(labels)) > 1:
        return calinski_harabasz_score(data, labels)
    return -1

# Visualizations

def plot_elbow_method(data, max_k=10):
    """
    Generate the Elbow Plot for K-Means.
    """
    inertias = []
    K_range = range(1, max_k + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, marker='o')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.show()

def plot_dendrogram(data):
    """
    Generate a Dendrogram for Hierarchical Clustering.
    """
    plt.figure(figsize=(10, 7))
    plt.title("Dendrogram")
    linked = linkage(data, method='ward')
    dendrogram(linked)
    plt.show()

def plot_pca_clusters(data, labels, title="PCA Clusters"):
    """
    Create a scatter plot of PCA components colored by cluster labels.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.show()
