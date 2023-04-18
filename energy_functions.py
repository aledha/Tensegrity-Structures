import numpy as np
np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)


def E(X, cables, bars, ms, consts):
    """ Energy of target function

    Args:
        X (array(3*N)): array of nodes
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        ms (array(N)): mass of nodes
        consts (list): constants g, rho, c, and k

    Returns:
        float: energy of system
    """
    g, rho, c, k = consts
    X = X.reshape(len(ms), 3)

    # External load
    E_ext = g * np.sum(ms * X[:, 2])

    # Bars
    bars_indices = np.asarray(np.where(bars != 0))
    grav_bar = 0
    elast_bar = 0
    for i, j in bars_indices.T:
        grav_bar += bars[i, j] * (X[i, 2] + X[j, 2])
        elast_bar += (np.linalg.norm(X[i] - X[j]) - bars[i, j])**2 / bars[i, j]**2
    E_bar = rho * g * grav_bar / 2 + c / 2 *  elast_bar

    # Cables
    cables_indices = np.asarray(np.where(cables != 0))
    elast_cable = 0
    for i, j in cables_indices.T:
        if np.linalg.norm(X[i] - X[j]) > cables[i, j]:
            elast_cable += (np.linalg.norm(X[i] - X[j]) - cables[i, j])**2 / cables[i, j]**2
    E_cable = elast_cable * k / 2
    return E_ext + E_bar + E_cable

def dE(Y, P, cables, bars, ms, N, M, consts):
    """ Compute gradient of target function. Only the indices corresponding to the variable nodes are returned.

    Args:
        Y (array(3*(N-M))): variable nodes
        P (array(3*M)): fixed nodes
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        ms (array(N)): mass of nodes
        N (int): number of nodes
        M (int): number of fixed nodes
        consts (list): constants g, rho, c, and k

    Returns:
        array(3*(N-M)): gradient of target function
    """
    Y = Y.reshape(N-M, 3)
    g, rho, c, k = consts

    # External loads
    dE_ext = np.zeros((N-M, 3))
    dE_ext[:, 2] = g * ms[M:]

    if M > 0:
        X = np.vstack((P, Y))
    else:
        X = Y

    # Bars
    bars_indices = np.asarray(np.where(bars != 0))
    grav_bar = np.zeros((N, 3))
    elast_bar = np.zeros((N, 3))
    
    for i, j in bars_indices.T:
        norm = np.linalg.norm(X[i] - X[j])
        grav_bar[i] += np.array([0, 0, bars[i, j]])
        grav_bar[j] += np.array([0, 0, bars[i, j]])
        elast_bar[i] += (norm - bars[i, j]) / (bars[i, j]**2 * norm) * np.array([X[i, 0] - X[j, 0], X[i, 1] - X[j, 1], X[i, 2] - X[j, 2]])
        elast_bar[j] += (norm - bars[i, j]) / (bars[i, j]**2 * norm) * np.array([-X[i, 0] + X[j, 0], -X[i, 1] + X[j, 1], -X[i, 2] + X[j, 2]])
    dE_bar = rho * g * grav_bar[M:] / 2 + c *  elast_bar[M:]

    # Cables
    cable_indices = np.asarray(np.where(cables != 0))
    elast_cable = np.zeros((N, 3))
    
    for i, j in cable_indices.T:
        norm = np.linalg.norm(X[i] - X[j])
        if norm > cables[i, j]:
            elast_cable[i] += (norm - cables[i, j]) / (cables[i, j]**2 * norm) * np.array([X[i, 0] - X[j, 0], X[i, 1] - X[j, 1], X[i, 2] - X[j, 2]])
            elast_cable[j] += (norm - cables[i, j]) / (cables[i, j]**2 * norm) * np.array([-X[i, 0] + X[j, 0], -X[i, 1] + X[j, 1], -X[i, 2] + X[j, 2]])
    dE_cable = k * elast_cable[M:]

    return (dE_ext + dE_bar + dE_cable).flatten()

def cmin(X):        
    """converting inequality constraint to equality constraints

    Args:
        X (3*N): takes in a flattend node matrix

    Returns:
        Y (3*N): a modified flattended matrix, where all positive values are set to zero
    """
    Y = X.copy()
    Y[Y > 0] = 0
    return Y

def Q(X, mu_1, mu_2, cables, bars, ms, consts):
    """ Modified energy function with a quadratic constraint terms

    Args:
        X (3*N): Node matrix
        mu_1 (float): penalty constant for constraints representing x^(i)_3 > 0
        mu_2 (float): penalty constant for constraints representing x^(1)_1=x^(1)_2=0
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        ms (array(N)): mass of nodes
        consts (list): constants g, rho, c, and k

    Returns:
        float: energy in system, in addition to penalty
    """
    return E(X, cables, bars, ms, consts) + 1/2 * mu_1 * np.sum(cmin(X[2::3])**2) + 1/2 * mu_2 * (X[0]**2 + X[1]**2)

def dQ(X, mu_1, mu_2, cables, bars, ms, consts, N):
    """The derivative of the modified energy function with a quadratic constraint term.

    Args:
        X (3*N): Node matrix
        mu_1 (float): penalty constant for constraints representing x^(i)_3 > 0
        mu_2 (float): penalty constant for constraints representing x^(1)_1=x^(1)_2=0
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        ms (array(N)): mass of nodes
        consts (list): constants g, rho, c, and k   
        N (int): number of nodes

    Returns:
        array(3*N): gradient of energy, in addition to penalty
    """
    barrier = np.zeros(X.shape).flatten()
    barrier[2::3] = mu_1 * cmin(X[2::3])
    barrier[0] = mu_2 * X[0]
    barrier[1] = mu_2 * X[1]
    return dE(X, np.array([]), cables, bars, ms, N, 0, consts) + barrier 