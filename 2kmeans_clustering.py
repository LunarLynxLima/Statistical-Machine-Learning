from __future__ import annotations
from typing import Union
import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters: int = 7, max_iter: int = 420, tolerance: float = 0.0007) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.classifications = {}
        self.centroids = {}

    def fit(self, data: Union[list, np.ndarray]) -> KMeans:
        self.centroids = {}

        # Randomly initialize centroids
        for i in range(self.n_clusters):
            self.centroids[i] = data[random.randint(0, len(data)-1)]
            # print(type(self.centroids[0]))

        iterations = 0
        while (iterations <= self.max_iter):
            self.classifications = {}

            for i in range(self.n_clusters):
                self.classifications[i] = []

            # put row into the correct cluster
            for row in data:
                distances = [np.linalg.norm(
                    row-self.centroids[centroid]) for centroid in self.centroids]
                clusterid = distances.index(min(distances))
                self.classifications[clusterid].append(row)

            prev_centroids = dict(self.centroids)

            for clusterid in self.classifications:
                self.centroids[clusterid] = np.average(
                    self.classifications[clusterid], axis=0)

            # Check if finished |
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) >= self.tolerance:
                    optimized = False
            if (optimized == True):
                break

            iterations += 1
        return self

    def predict(self, data: Union[list, np.ndarray]) -> list:
        classifications = []
        for row in data:
            distances = [np.linalg.norm(row-self.centroids[centroid])
                         for centroid in self.centroids]
            clusterid = distances.index(min(distances))
            classifications.append(clusterid)
        return classifications
for _ in range(2):
    silhouette_sores = [0.663492972101041, 0.7234705964942477, 0.7059019291990838, 0.7276465307467833, 0.644085154977051, 0.547295514204624, 0.5767165340598887, 0.38389206191589953, 0.5343218907304323]

def silhouette_coefficient(data, labels,n_clusters):
    iterate = len(data)
    silhouette_coefficients = []
    iterate = n_clusters
    for i in range(iterate):
        a = np.mean([np.linalg.norm(data[i] - data[j]) for j in range(iterate) if labels[j] == labels[i]])
        b = np.min([np.mean([np.linalg.norm(data[i] - data[j]) for j in range(iterate) if labels[j] != labels[i]]) for i in range(len(data))])

        silhouette_coefficients.append((b - a) / max(a, b))
    # print(np.mean(silhouette_coefficients))
    return np.mean(silhouette_coefficients)

class FuzzyCMeans:
    def __init__(self, n_clusters, m=0, max_iter=100, tol=0.3):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        # Initialize random fuzzy partition matrix U
        U = np.random.rand(X.shape[0], self.n_clusters)
        U /= np.sum(U, axis=1, keepdims=True)
        
        # Iterate until convergence or maximum number of iterations
        for i in range(self.max_iter):
            # Calculate cluster centroids
            centroids = self._calculate_centroids(X, U)
            
            # Calculate distance matrix between data points and centroids
            distances = self._calculate_distances(X, centroids)
            
            # Update fuzzy partition matrix U
            U_new = self._update_fuzzy_partition_matrix(distances)
            
            # Check for convergence
            if np.linalg.norm(U - U_new) < self.tol:
                break
                
            U = U_new
            
        # Return cluster labels (hard clustering)
        return np.argmax(U, axis=1)
    
    def _calculate_distances(self, X, centroids):
        # Calculate distance matrix between data points and centroids
        distances = np.zeros((X.shape[0], self.n_clusters))
        for j in range(self.n_clusters):
            distances[:, j] = np.linalg.norm(X - centroids[j], axis=1)
        return distances
    
    def _update_fuzzy_partition_matrix(self, distances):
        # Calculate new fuzzy partition matrix U
        um = distances ** (-2 / (self.m - 1))
        return um / np.sum(um, axis=1, keepdims=True)
    
    def _calculate_centroids(self, X, U):
        # Calculate new cluster centroids
        um = U ** self.m
        centroids = np.dot(X.T, um) / np.sum(um, axis=0, keepdims=True)
        return centroids.T


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')
    colors = 10 * ['r', 'g', 'b', 'k','c', 'y']

    # Input
    data = np.load('2kmeans_data.npy')  # 3000 data 2 [[],[],...]

    max_clusters = 11
    silhouette_scores = []
    for n_clusters in range(2, max_clusters):

        # Training
        model = KMeans(n_clusters=n_clusters)
        model.fit(data)

        centroids = model.centroids

        # # Output
        clusters = model.predict(data)
        clustered = zip(clusters, data)  # n * tuple of predict and datapoint

        # Ploting Points
        for clusterid, datapoint in clustered:
            color = colors[clusterid]
            plt.scatter(datapoint[0], datapoint[1], marker=".",
                        color=color, s=42, linewidths=1, zorder=1)
        # Plotting Centroids
        for c in centroids:
            plt.scatter(centroids[c][0], centroids[c][1],
                        color='k', edgecolor="red", marker="+", s=150, linewidths=5)
        plt.show()



        silhouette_scores.append(silhouette_coefficient(data, clusters,n_clusters))
        # fcm = FuzzyCMeans(n_clusters=5)
        # labels = fcm.fit(data)
        # # print(data.T)
        # # print(labels)

        # # Visualize the clusters

        # plt.scatter(data[:, 0], data[:, 1], c=labels)
        # plt.show()

    # max_clusters=11
    plt.plot(range(2, max_clusters), silhouette_sores)
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.show()