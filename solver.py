import numpy as np
import custom_plot_structures as cplot

np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)

def line_search(x, p, c1, c2, f, df, maxiter = 100):
    """ Line search using Wolfe conditions

    Args:
        x (array): initial vector
        p (array): search direction
        c1, c2 (ints): backtracking parameters
        f (func): target function
        df (func): gradient of target function
        maxiter (int, optional): maximum number of iterations. Defaults to 100.

    Returns:
        alpha (int): step length
    """
    assert(c1 < c2)
    alpha_min, alpha_max = 0, np.inf
    alpha = 1
    I = f(x + alpha * p) <= f(x) + c1 * alpha * np.inner(df(x), p)                      # Armijo condition
    II = np.abs(np.inner(df(x + alpha * p), p)) <= c2 * np.abs(np.inner(df(x), p))      # Curvature condition
    iter = 0
    while not (I and II):
        if iter > maxiter:
            break
        if not I:
            alpha_max = alpha
            alpha = (alpha_min + alpha_max) / 2
        else:
            alpha_min = alpha
            if alpha_max < np.inf:
                alpha = (alpha_min + alpha_max) / 2
            else:
                alpha *= 2
        I = f(x + alpha * p) <= f(x) + c1 * alpha * np.inner(df(x), p)
        II = np.abs(np.inner(df(x + alpha * p), p)) <= c2 * np.abs(np.inner(df(x), p))
        iter += 1
    return alpha

def BFGS(X0, N, M, cables, bars, maxiter, f, df, tol = 1e-6, keep_limits = [False, False, False]):
    """
    BFGS algorithm to find a local minimiser of a tensegrity system, using fixed nodes. Also plots the solution

    Args:
        X0 (array(N, 3)): x,y,z coordinates of nodes
        N (int): number of nodes
        M (int): number of fixed nodes
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        maxiter (int): number of iteriations
        f (func): function to minimize
        df (func): gradient of f
        keep_limits (list of bool, optional): Keep limits between plots in x, y, and z axis respectively

    Returns:
        (array(N, 3)): solution
    """
    original_limits = cplot.plot_points(X0.reshape((N,3)), bars, cables, M, title = 'Original system')

    P = X0[:3 * M]                      # Fixed nodes
    Y0 = X0[3 * M:]                     # Variable nodes

    c1, c2 = 0.025, 0.2                 # Backtracking parameters

    # First: Gradient descent
    p = -df(Y0)
    alpha = line_search(Y0, p, c1, c2, f, df)
    Y1 = Y0 + alpha * p

    sk = Y1 - Y0
    yk = df(Y1) - df(Y0)
    sk = np.array([sk]).reshape((3*(N-M), 1))
    yk = np.array([yk]).reshape((3*(N-M), 1))
    H = (sk.T @ yk) / (yk.T @ yk) * np.eye(3 * (N - M))

    for k in range(maxiter):
        p = -H @ df(Y1)

        alpha = line_search(Y1, p, c1, c2, f, df)
        Y0 = Y1
        Y1 = Y0 + alpha * p
        
        sk = Y1 - Y0
        yk = df(Y1) - df(Y0)

        sk = np.array([sk]).reshape((3*(N-M), 1))
        yk = np.array([yk]).reshape((3*(N-M), 1))

        Hkyk = H@yk
        Sk = 1 / (yk.T @ sk)
        H = H.copy() - Sk * (sk @ Hkyk.T + Hkyk @ sk.T) + sk @ sk.T * (Sk**2 * Hkyk.T @ yk + Sk)

        if M == 0:              # Fix coordinate system to the first node
            Y1[0::3] -= Y1[0]
            Y1[1::3] -= Y1[1]

        if np.linalg.norm(sk) < tol or np.linalg.norm(yk) < tol:
            print("Converged after", k, "iterations")
            break
    
    for i in range(3):
        if not keep_limits[i]:
            original_limits[i] = None
    
    cplot.plot_points((np.concatenate((P, Y1))).reshape(N, 3), bars, cables, M, lims = original_limits)
    return Y1