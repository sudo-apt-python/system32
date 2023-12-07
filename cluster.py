import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_std = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_std)
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_std)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_std)

def plot_clusters(labels, method):
    plt.scatter(X_std[:, 0], X_std[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.title(f'Clustering Result ({method})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


plot_clusters(kmeans_labels, 'KMeans')
plot_clusters(hierarchical_labels, 'Hierarchical Clustering')
plot_clusters(gmm_labels, 'Gaussian Mixture Model')
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
themdbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = themdbscan.fit_predict(X_std)
plot_clusters(dbscan_labels, 'DBSCAN')
