import numpy as np
import pandas as pd
import pprint as prt
import matplotlib.pyplot as plt
import math

def generate():
    # generate a sinusoidal wave
    t = np.linspace(0, 2*np.pi, 1000)  # time vector
    f = 2*np.pi  # frequency in Hz
    A = 1  # amplitude
    sin_wave = A * np.sin(2*np.pi*f*t)

    # generate a ramp wave
    ramp_wave = np.linspace(0, 7, len(t))
    return t, sin_wave, ramp_wave

def plottwo(t, sin_wave, ramp_wave):
    # t,sin_wave, ramp_wave = generate()
    # plot the waves
    plt.plot(t, sin_wave)
    plt.plot(t, ramp_wave)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()
    return sin_wave, ramp_wave

def plot(lis: list[float]):
    t = np.linspace(0, 2*np.pi, 1000)
    plt.plot(t, lis, label='Signal')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend()
    plt.show()

class ICA:
    # def __init__(this, sampledata: list[list[float]]) -> None:
    #     this.sampledata = sampledata

    def __eigen(covmat: list[list[float]]):
        covmat = np.nan_to_num(covmat, nan=1e-15)
        return np.linalg.eig(covmat)

    def __centering(rawdata):
        rawdata = np.array(rawdata)
        mean = rawdata.mean(axis=1, keepdims=True)
    # def __centering(rawdata : list[list[float]]):
    #     return (rawdata - np.mean(rawdata, axis=0))

        return (rawdata - mean)

    def __whitening(rawdata):
        rawdata = ICA.__centering(rawdata)
        cov = np.cov(rawdata)
        eigenvalues, E = np.linalg.eigh(cov)
        D = np.diag(eigenvalues)
        inversesqrtD = np.sqrt(np.linalg.inv(D))
        whiteneddata = np.dot(E, np.dot(inversesqrtD, np.dot(E.T, rawdata)))

        return whiteneddata

    def __g(x):
        return np.tanh(x)

    def __gderivative(x):
        return 1 - ICA.__g(x) * ICA.__g(x)

    def __recalutatew(wi, data):
        neww = (data * ICA.__g(np.dot(wi.T, data))).mean(axis=1) - \
            ICA.__gderivative(np.dot(wi.T, data)).mean() * wi
        neww /= np.sqrt((neww ** 2).sum())
        return neww

    def computeingWandoutput(rawdata, iterations=1000, tolerance=0.01):
        whiteneddata = ICA.__whitening(rawdata)

        features = whiteneddata.shape[0]
        W = np.zeros((features, features), dtype=whiteneddata.dtype)
        for i in range(features):

            wi = np.random.rand(features)

            for j in range(0,iterations):

                neww = ICA.__recalutatew(wi, whiteneddata)

                if i >= 1:
                    neww -= np.dot(np.dot(neww, (W[:i]).T), W[:i])

                distance = np.abs(np.abs((wi * neww).sum()) - 1)

                wi = neww

                if distance < tolerance:
                    break

            W[i, :] = wi

        ans = np.dot(W, whiteneddata)

        return ans

## ---------------------------------------- Q4 ---------------------------------------- ##
if __name__ == "__main__":

    # Step 1.1 : Generate a sinusoidal wave and a ramp wave.
    time, sinwave, rampwave = generate()
    # Step 1.2 : Plot them.
    # plottwo(time, sinwave, rampwave)

    # Step2.1 : Mixing Signal
    stackedsignal = np.vstack((sinwave, rampwave)).T
    mixingmatrix = [[0.5, 1], [1, 0.5]]
    mixsignal = ((stackedsignal.dot(mixingmatrix)).T)
    # Step2.2 : Plot the mixed signals.
    # plottwo(time, mixsignal[0], mixsignal[1])

    # Step3.1 : ICA to recover the original signals
    recoveredsignal = ICA.computeingWandoutput(mixsignal)
    # Step3.2 : Plot recovered signals
    plottwo(time, recoveredsignal[0], recoveredsignal[1])