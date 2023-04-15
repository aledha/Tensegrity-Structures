import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

import custom_plot_structures as cplot
import energy_functions as efunc
import solver as solver

np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)

def cable_sys():
    g, rho, c, k = 9.81, 1, 1, 3
    consts = [g, rho, c, k]
    N = 8
    M = 4
    P = np.array([[5, 5, 0],
                [-5, 5, 0],
                [-5, -5, 0],
                [5, -5, 0]])
    Y0 = (np.random.rand(N-M, 3) - 0.5)*1000
    X0 = np.vstack((P, Y0))

    cables = np.zeros((N, N))
    bars = np.zeros((N, N))
    cables[0, 4], cables[1, 5], cables[2, 6], cables[3, 7], cables[4, 5], cables[4, 7], cables[5, 6], cables[6, 7] = [3] * 8

    ms = np.ones(N) / (6 * g)

    def f(Y):
        X = np.concatenate((P.flatten(), Y))
        return efunc.E(X, cables, bars, ms, consts)

    def df(Y):
        return efunc.dE(Y, P, cables, bars, ms, N, M, consts)

    Y = solver.BFGS(X0.flatten(), N, M, cables, bars, 100, f, df)
    print(Y.reshape((N-M, 3)))
    