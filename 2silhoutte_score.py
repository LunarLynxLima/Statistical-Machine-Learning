from __future__ import annotations
from typing import Union
import numpy as np
import random
#Manhattan Distance
def distance(x,y):
  return np.sqrt(np.sum((x-y)**2))
  # return np.sum(x-y)
class silhouette_analysis:
    def __manhattandistance(p1: list[float], p2: list[float]) -> float:
        return -1 if (len(p1) != len(p2)) else sum([abs(p1[i]-p2[i]) for i in range(len(p1))])

    def __euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    def silhouette_score(X, labels,n_clusters):
        # n_clusters = len(set(labels))
        cluster_centers = np.array([np.mean(X[labels == i], axis=0) for i in range(n_clusters)])
        scores = np.zeros(len(X))

        for i in range(len(X)):
            a_i = np.mean([distance(X[i], X[j]) for j in range(len(X)) if labels[j] == labels[i] and i != j])
            b_i = np.min([np.mean([distance(X[i], X[j]) for j in range(len(X)) if labels[j] == k]) for k in range(n_clusters) if k != labels[i]])
            if(max(b_i, a_i) > 0):
                scores[i] = (b_i - a_i) / max(b_i, a_i) 
            else:
                scores[i] =  0

        return np.mean(scores)