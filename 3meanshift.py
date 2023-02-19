import numpy as np
import pandas as pd
import pprint as prt
import matplotlib.pyplot as plt
import numpy.random as rnd
import matplotlib.pyplot as plotter
from PIL import Image
from typing import Union, Optional


def extractjpgs(picpath: str) -> list[list[int]]:
    # data = []
    var = Image.open(picpath)
    arr = np.asarray(var)
    data = np.reshape(arr, (arr.shape[0] * arr.shape[1], arr.shape[2]))
    return arr


class Meanshift:
    # def __init__(self, bandwidth: float = 0.2, iterate: int = 7):
    #     self.bandwidth = bandwidth
    #     self.iterate = iterate

    def __manhattandistance(p1: list[float], p2: list[float]) -> float:
        return -1 if (len(p1) != len(p2)) else sum([abs(p1[i]-p2[i]) for i in range(len(p1))])

    def __euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))


def normalize(data: list[list[float]]):
    data = np.array(data)
    max: list[int] = (np.amax(data, axis=0))
    return data/max


def mean_shift(data, bandwidth=50):
    centroids = data.copy()

    while True:
        euclidean = np.sqrt(
            np.sum((centroids[:, np.newaxis] - centroids) ** 2, axis=2))

        weights = np.exp(-0.5 * (euclidean / bandwidth) ** 2)

        shiftmean = np.sum(
            weights[:, :, np.newaxis] * (centroids[:, np.newaxis] - centroids), axis=1)
        shift = np.sqrt(np.sum(shiftmean ** 2, axis=1))

        recentroids = centroids + shiftmean

        shifts_norms_new = np.sqrt(
            np.sum((recentroids - centroids) ** 2, axis=1))

        if np.max(shifts_norms_new) < 1e-6:
            break

        centroids = recentroids

    euclidean = np.sqrt(np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2))
    labels = np.argmin(euclidean, axis=1)

    return centroids, labels


## ---------------------------------------- Q3 ---------------------------------------- ##
if __name__ == "__main__":
    data = extractjpgs(r"D:\OneDrive\Desktop\peppers.png")
    normie = normalize(data)  # 196608*3
    centroids, label = mean_shift(normie, 3)
    print(label)
    tmp = Image.fromarray(data)
    tmp.save("output.jpg")
    # data = [[0, 1], [2, 3]]
    # print(len(data[0]),len(data))
