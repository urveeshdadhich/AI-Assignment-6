import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def initialize_centroids(X, k):
    """Initialize k centroids randomly"""
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    return X[random_indices]

def assign_clusters(X, centroids):
    """Assign each point to nearest centroid"""
    distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def update_centroids(X, cluster_labels, k):
    """Update centroids based on mean of points in each cluster"""
    centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])
    return centroids

def kmeans(X, k, max_iters=100):
    """K-means clustering algorithm"""
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    
    for _ in range(max_iters):
        # Assign clusters
        old_cluster_labels = assign_clusters(X, centroids)
        
        # Update centroids
        centroids = update_centroids(X, old_cluster_labels, k)
        
        # Check for convergence
        new_cluster_labels = assign_clusters(X, centroids)
        if np.all(old_cluster_labels == new_cluster_labels):
            break
    
    return centroids, new_cluster_labels

def plot_clusters(X, labels, centroids, k, title):
    """Plot the clusters and centroids"""
    plt.figure(figsize=(10, 6))
    
    # Plot points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                c='red', marker='x', s=200, linewidths=3, 
                label='Centroids')
    
    plt.title(title)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.show()

# Load and preprocess data
data = pd.read_csv('kmeans_blobs.csv')
X = data[['x1', 'x2']].values

# Normalize the data
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Set random seed for reproducibility
np.random.seed(42)

# Run k-means for k=2
centroids_k2, labels_k2 = kmeans(X_normalized, k=2)
plot_clusters(X_normalized, labels_k2, centroids_k2, k=2, 
             title='K-means Clustering (k=2)')

# Run k-means for k=3
centroids_k3, labels_k3 = kmeans(X_normalized, k=3)
plot_clusters(X_normalized, labels_k3, centroids_k3, k=3, 
             title='K-means Clustering (k=3)')
