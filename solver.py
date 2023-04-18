import numpy as np
import custom_plot_structures as cplot

np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)

def line_search(x, p, c1, c2, f, df, maxiter = 100):
    """ Line search using strong Wolfe conditions

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
    I = f(x + alpha * p) <= f(x) + c1 * alpha * p.T @ df(x)                     # Armijo condition
    II = np.abs(p.T @ df(x + alpha * p)) <= c2 * np.abs(p.T @ df(x))            # Curvature condition
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
        I = f(x + alpha * p) <= f(x) + c1 * alpha * p.T @ df(x)                     # Armijo condition
        II = np.abs(p.T @ df(x + alpha * p)) <= c2 * np.abs(p.T @ df(x))            # Curvature condition
        iter += 1
    return alpha

def BFGS(X0, N, M, maxiter, f, df, tol = 1e-6):
    """
    BFGS algorithm to find a local minimiser of a tensegrity system, either with fixed nodes or ground constraint.

    Args:
        X0 (array(N, 3)): x,y,z coordinates of nodes
        N (int): number of nodes
        M (int): number of fixed nodes
        maxiter (int): number of iteriations
        f (func): function to minimize
        df (func): gradient of f
        tol (float): tolerance for when to stop iteration. Depends on the difference in the node positions.

    Returns:
        (array(N, 3)): solution
    """

    P = X0[:3 * M]                      # Fixed nodes
    Y0 = X0[3 * M:]                     # Variable nodes

    c1, c2 = 0.02, 0.2                 # Backtracking parameters

    # First: One step of gradient descent
    grad = df(Y0)
    p = -grad
    alpha = line_search(Y0, p, c1, c2, f, df)
    Y1 = Y0 + alpha * p
    grad_new = df(Y1)

    sk = Y1 - Y0
    yk = grad_new - grad
    sk = np.array([sk]).reshape((3*(N-M), 1))
    yk = np.array([yk]).reshape((3*(N-M), 1))
    H = (sk.T @ yk) / (yk.T @ yk) * np.eye(3 * (N - M))

    gradients = np.zeros(maxiter)       # Array to store the norm of gradients
    for k in range(maxiter):
        gradients[k] = np.linalg.norm(grad)

        grad = grad_new
        p = -H @ grad

        alpha = line_search(Y1, p, c1, c2, f, df)
        Y0 = Y1
        Y1 = Y0 + alpha * p
        
        grad_new = df(Y1)

        sk = Y1 - Y0
        yk = grad_new - grad

        sk = np.array([sk]).reshape((3*(N-M), 1))
        yk = np.array([yk]).reshape((3*(N-M), 1))

        Hkyk = H@yk
        Sk = 1 / (yk.T @ sk)
        H = H.copy() - Sk * (sk @ Hkyk.T + Hkyk @ sk.T) + sk @ sk.T * (Sk**2 * Hkyk.T @ yk + Sk)

        if np.linalg.norm(sk) < tol:
            gradients[k+1] = np.linalg.norm(grad_new)
            gradients = gradients[:k+2]
            print("Converged after", k, "iterations")
            break

    X1 = (np.concatenate((P, Y1)))              # Reassembling the fixed points
    return X1, gradients