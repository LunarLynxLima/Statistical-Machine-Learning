import numpy as np
import pandas as pd
import pprint as prt
import matplotlib.pyplot as plt
import numpy.random as rnd
import matplotlib.pyplot as plotter
import os
from PIL import Image


def extractjpgs(folderpath: str, exceptions=90) -> list[list[int]]:
    data = []
    i = countexceptions = 1
    bol = True
    while (bol):
        path = folderpath + "\img_" + str(i) + ".jpg"
        try:
            var = Image.open(path)
            arr = np.asarray(var)
            data.append(arr.flatten())
            i += 1
        except:
            if (countexceptions > exceptions):
                bol = False
            else:
                countexceptions += 1
                i += 1
    return data


class PCA:
    # def __init__(this, sampledata: list[list[float]]) -> None:
    # this.sampledata = sampledata
    def standardization(rawdata: list[list[float]]):
        sd = (np.std(rawdata, axis=0, ddof=1))
        epsilon = 1e-15
        mask = np.isclose(sd, 0)
        sd = np.where(mask, epsilon, sd)
        # ddof =1 [== sample SD]
        return (rawdata - np.mean(rawdata, axis=0))/sd

    def __covariance_matrix(standarddata: list[list[float]]):
        return np.cov(standarddata.T, ddof=0)

    def __eigen(covmat: list[list[float]]):
        covmat = np.nan_to_num(covmat, nan=1e-15)
        return np.linalg.eig(covmat)

    def __order_eigen(eigenvectors: list[list[float]], eigenvalues: list[float]):
        idx = eigenvalues.argsort()[::-1]
        sortedeigenvalues = eigenvalues[idx]
        sortedeigenvectors = [[] for _ in range(len(idx))]

        for i in range(0, len(idx)):
            sortedeigenvectors[i] = eigenvectors[idx[i]]

        return sortedeigenvectors, sortedeigenvalues

    def __feature_vector(data: list[list[float]]):
        covmat = PCA.__covariance_matrix(data)
        eigen_value, eigen_vector = PCA.__eigen(covmat)
        sorted_eigen_vectors, sorted_eigen_values = PCA.__order_eigen(
            eigen_vector.T, eigen_value)
        return sorted_eigen_vectors, sorted_eigen_values

    def feature_reduction_and_reducedmatrix(data: list[list[float]], featuresrequired: int = 7):
        data = PCA.standardization(data)
        featurevector, eigenvalues = PCA.__feature_vector(data)
        reducedMatrix = [[] for _ in range(featuresrequired)]
        sumof_n_eigenvalues = 0
        for i in range(featuresrequired):
            reducedMatrix[i] = featurevector[i]
            sumof_n_eigenvalues += eigenvalues[i]
        explainedvariance = sumof_n_eigenvalues/sum(eigenvalues)
        # print(explainedvariance)
        a = np.array(reducedMatrix)

        return a.T, explainedvariance

    def plotexplainedvariance(data: list[list[float]]):
        allexplainedvariances = [[0], [0]]
        covereightyPC = 0

        for i in range(1, len(data)):
            print('PCs = ', i)
            D, explainedvariance = PCA.feature_reduction_and_reducedmatrix(
                data, featuresrequired=i)
            allexplainedvariances[0].append(explainedvariance)
            allexplainedvariances[1].append(i)
            if (explainedvariance >= 0.80 and covereightyPC == 0):
                covereightyPC = i
                color = ['red' for _ in range(covereightyPC)]
        # colour2 = ['green' for _ in range(covereightyPC,len(data))]
        color = color + ['green' for _ in range(covereightyPC, len(data))]
        fig, ax = plt.subplots()
        ax.axhline(y=0.8, color='r', label="Horizontal line")
        # plotter.plot(allexplainedvariances[1],allexplainedvariances[0])
        plotter.bar(allexplainedvariances[1],
                    allexplainedvariances[0], color=color)
        plotter.show()
        return allexplainedvariances, covereightyPC

if __name__ == "__main__":
    path = r"D:\OneDrive\Desktop\archive\trainingSample\trainingSample"
    

    folder_path = path
    items = os.listdir(folder_path)
    folders = [f for f in items if os.path.isdir(os.path.join(folder_path, f))]
    data=[]
    for i in folders:
        print(type(i),i)
        tmp = extractjpgs(path+"\\"+i)
        data.append(tmp)
    # Print the list of folders
    print(len(data[0])) #10*

    # data = [[1,5,3,1],[4,2,6,3],[1,4,3,2],[4,4,1,1],[5,5,2,3]]
    featuresrequired = 5
    # featuresrequired = 25
    # featuresrequired = 125
    D, explainedvariance = PCA.feature_reduction_and_reducedmatrix(
        data, featuresrequired)
    reduceddata = np.dot(data, D)
    # print((D))


    # e) Plot explained-variance
    allexplainedvariances, covereightyPC = PCA.plotexplainedvariance(
        data)  # allexplainedvariances -> [[],[]...]
