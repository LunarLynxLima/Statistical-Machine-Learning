import numpy as np
import pandas as pd
import pprint as prt
import matplotlib.pyplot as plt
import numpy.random as rnd
import statistics as stat
import matplotlib.pyplot as plotter
from PIL import Image
class point:
    def __init__(self,list):
        self.list =list

class kNN:
    def __init__(this, sampledata: list[point], k: int) -> None:
        this.sampledata = sampledata
        this.k = k
    def __manhattandistance(p1: point, p2: point) -> float:
        # points are not in same dimension
        return -1 if (len(p1) != len(p2)) else sum([abs(p1[i]-p2[i]) for i in range(len(p1))])
        # if (len(p1) != len(p2)):
        #     print("points are not in same dimension")
        #     return -1
        # return sum([abs(p1[i]-p2[i]) for i in range(len(p1))])
    def __orderlistasperotherlist(orderthislist: list[point], byorder: point):
        byorder = np.array(byorder)
        idx = byorder.argsort()
        sortedby = byorder[idx]
        sortedlist = [[] for _ in range(len(idx))]
        for i in range(0, len(idx)):
            sortedlist[i] = orderthislist[idx[i]]

        return sortedlist, sortedby
    def __allkNN(point: point, allpoints: list[point]) -> list[point]:
        # neighbours = [[] for _ in range(k)]
        allpointsdistance = [[] for _ in range(len(allpoints))]
        for i in range(len(allpoints)):
            allpointsdistance[i] = kNN.__manhattandistance(point, allpoints[i])
        neighbourssorted, allpointsdistance = kNN.__orderlistasperotherlist(
            allpoints, allpointsdistance)
        return neighbourssorted

    def kNN(point: point, orderedneighbour: list[point], k: int = 7) -> list[point]:
        orderedneighbour = kNN.__allkNN(point, orderedneighbour)
        return orderedneighbour[0:k]

    def approximateans(knearestpoints:list[point],legendindex = 0) -> int:
        legend = [knearestpoints[i][legendindex] for i in range(len(knearestpoints))]
        # print(legend)
        ans = stat.mode(legend)
        return legend[ans]

#main
if __name__ == "__main__":
    knn = (kNN.kNN([1,0, 0], [[0, 0,0],[0,0,0],[2, 2,1], [5, 5,5]], 3))
    print((knn))