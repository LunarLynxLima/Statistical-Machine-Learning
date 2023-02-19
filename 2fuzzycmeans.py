from __future__ import annotations
from typing import Union
import numpy as np
import random

class FuzzyCMeans:
    def __init__(self, numclusters, m=2, maxiteration=100, beta=0.3):
        self.numclusters = numclusters
        self.m = m
        self.maxiteration = maxiteration
        self.beta = beta

    def __calculate_distances(self, X, centroids):
        # Calculate distance matrix between data points and centroids
        pointtocentroid = np.zeros((X.shape[0], self.numclusters))
        for j in range(self.numclusters):
            pointtocentroid[:, j] = np.linalg.norm(X - centroids[j], axis=1)
        return pointtocentroid

    def __update_fuzzy_partition_matrix(self, pointtocentroid):
        # Calculate new fuzzy partition matrix U
        um = pointtocentroid ** (-2 / (self.m - 1))
        return um / np.sum(um, axis=1, keepdims=True)

    def __calculate_centroids(self, X, U):
        # Calculate new cluster centroids
        um = U ** self.m
        centroids = np.dot(X.T, um) / np.sum(um, axis=0, keepdims=True)
        return centroids.T

    def fit(self, X):
        # Initialize random fuzzy partition matrix U
        U = np.random.rand(X.shape[0], self.numclusters)
        U /= np.sum(U, axis=1, keepdims=True)

        # Iterate until (convergence || maximum iterations)
        i = 0
        while(i<=self.maxiteration):

            # Calculate cluster centroids
            centroids = self.__calculate_centroids(X, U)

            # Calculate distance matrix between data points and centroids
            pointtocentroid = self.__calculate_distances(X, centroids)

            # Update fuzzy partition matrix U
            neU = self.__update_fuzzy_partition_matrix(pointtocentroid)

            # Check for convergence
            if np.linalg.norm(U - neU) < self.beta:
                break

            # Update U
            U = neU
            i+=1
            J = np.sum(U**self.m * np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)**2)
            print(J)
        # Return cluster labels (1 point 1 Cluster policy)
        return U ,J


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import style
    style.use('ggplot')
    colors = 10 * ['r', 'g', 'b', 'k', 'c', 'y']

    # Input
    # 3000 data 2 [[],[],...]    fcm = FuzzyCMeans(numclusters=5)
    data = np.load('2kmeans_data.npy')
    # for i in range(2,10):
    numclusters=5
    fcm = FuzzyCMeans(numclusters)
    U,J = fcm.fit(data)
    print('number of clusters = ',numclusters,'|  J(min) = ',J)
    singlelabel = np.argmax(U, axis=1)

    # Visualize the clusters
    plt.scatter(data[:, 0], data[:, 1], c=singlelabel)
    plt.show()
