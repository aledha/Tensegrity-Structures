import numpy as np
import energy_functions as efunc
import solver as solver
import custom_plot_structures as cplot

np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)

def cable_sys(rho = 1, c = 1, k = 3, N = 8, M = 4, max_iter = 100):
    """ Example of a system of cables (a net) and points. Some of the points are fixed. The solution is known."""
    g = 9.81
    consts = [g, rho, c, k]
    P = np.array([[5, 5, 0],
                [-5, 5, 0],
                [-5, -5, 0],
                [5, -5, 0]])
    Y0 = (np.random.rand(N-M, 3) - 0.5)*1000
    X0 = np.vstack((P, Y0)).flatten()

    cables = np.zeros((N, N))
    bars = np.zeros((N, N))

    cables[0, 4], cables[1, 5], cables[2, 6], cables[3, 7], cables[4, 5], cables[4, 7], cables[5, 6], cables[6, 7] = [3] * 8

    ms = np.ones(N) / (6 * g)

    def f(Y):
        X = np.concatenate((P.flatten(), Y))
        return efunc.E(X, cables, bars, ms, consts)

    def df(Y):
        return efunc.dE(Y, P, cables, bars, ms, N, M, consts)

    X1, gradients = solver.BFGS(X0, N, M, max_iter, f, df)
    cplot.plot_points(X0, X1, cables, bars, M, gradients, title = 'Cable system')
    print(X1.reshape(N, 3))
    
def cables_and_bars(g = 0, rho = 0, c = 1, k = 0.3, N = 8, M = 4, max_iter = 1000):
    """ A more complex test case with both cables and bars, with a known solution. """
    consts = [g, rho, c, k]
    P = np.array([[1, 1, 0],
                [-1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0]])
    Y0 = np.random.rand(N-M, 3) - 0.5
    Y0[:, 2] = 4
    P[0, 2] = 0.5
    X0 = np.vstack((P, Y0)).flatten()

    cables = np.zeros((N, N))
    bars = np.zeros((N, N))

    bars[0, 4], bars[1, 5], bars[2, 6], bars[3, 7] = [10] * 4
    cables[0, 7], cables[1, 4], cables[2, 5], cables[3, 6] = [8] * 4
    cables[4, 5], cables[4, 7], cables[5, 6], cables[6, 7] = [1] * 4

    ms = np.zeros(N)

    def f(y):
        X = np.concatenate((P.flatten(), y))
        return efunc.E(X, cables, bars, ms, consts)

    def df(y):
        return efunc.dE(y, P, cables, bars, ms, N, M, consts)

    X1, gradients = solver.BFGS(X0, N, M, max_iter, f, df)
    cplot.plot_points(X0, X1, cables, bars, M, gradients, title = 'System with cables and bars')
    print(np.float16(X1.reshape(N, 3)))

def tensegrity_table(g = 9.81, rho = 0, c = 10, k = 10, N = 10, M = 5, max_iter = 100):
    """ Construction, simulation and plotting of a so-called tensegrity table."""
    consts = [g, rho, c, k]
    P = np.array([[1, 1, 0],
                [-1, 1, 0],
                [-1, -1, 0],
                [1, -1, 0],
                [0, 0, 1]])
    Y0 = np.random.rand(N-M, 3) - 0.5
    Y0 = np.copy(P)
    Y0[:, 2] += 4
    Y0[-1, 2] = np.sqrt(2) - 1

    X0 = np.vstack((P, Y0)).flatten()

    cables = np.zeros((N, N))
    bars = np.zeros((N, N))

    bars[0, 1], bars[1, 2], bars[2, 3], bars[0, 3] = [2] * 4
    bars[0, 4], bars[1, 4], bars[2, 4], bars[3, 4] = [np.sqrt(3)] * 4

    bars[5, 6], bars[6, 7], bars[7, 8], bars[5, 8] = [2] * 4
    bars[5, 9], bars[6, 9], bars[7, 9], bars[8, 9] = [np.sqrt(3)] * 4

    cables[0, 5], cables[1, 6], cables[2, 7], cables[3, 8] = [2] * 4
    cables[4, 9] = 2*np.sqrt(2) - 2

    ms = 0.001 * np.ones(N)

    def f(Y):
        X = np.concatenate((P.flatten(), Y))
        return efunc.E(X, cables, bars, ms, consts)

    def df(Y):
        return efunc.dE(Y, P, cables, bars, ms, N, M, consts)

    X1, gradients = solver.BFGS(X0, N, M, max_iter, f, df)
    cplot.plot_points(X0, X1, cables, bars, M, gradients, title = 'Tensegrity table', keep_zlim = True, plot_view='x')

def with_ground_quad_constraints():


    g, rho, c, k = 9.81, 0.1, 100000, 1000
    consts = [g, rho, c, k]
    N = 8
    M = 0
    P = np.array([[1, 1, 0.1],
                [-1, 1, 0.1],
                [-1, -1, 0.1],
                [1, -1, 0.1]])

    s, t = 0.70970, 9.54287
    Ystar = np.array([[-s,0,t], 
                [0, -s, t],
                [s,0,t],
                [0,s,t]])

    Y0 = Ystar

    X0 = np.vstack((P, Y0)).flatten()
    X0[0::3] -= X0[0]        # Moving structure such that the 
    X0[1::3] -= X0[1]        # first nodes x,y values are 0

    cables = np.zeros((N, N))
    bars = np.zeros((N, N))

    cables[0, 1], cables[1, 2], cables[2, 3], cables[0, 3] = [2] * 4
    bars[0, 4], bars[1, 5], bars[2, 6], bars[3, 7] = [10] * 4
    cables[0, 7], cables[1, 4], cables[2, 5], cables[3, 6] = [8] * 4
    cables[4, 5], cables[4, 7], cables[5, 6], cables[6, 7] = [1] * 4

    ms = np.zeros(N) * 0.001

    mu_1 = 100000
    mu_2 = 0.001

    def f(X):
        return efunc.Q(X, mu_1, mu_2, cables, bars, ms, consts)

    def df(X):
        return efunc.dQ(X, mu_1, mu_2, cables, bars, ms, consts, N)

    X1, gradients = solver.BFGS(X0, N, M, 1000, f, df, tol = 1e-6)
    cplot.plot_points(X0, X1, cables, bars, M, gradients, title = 'Free standing system', zprojection = False)
    print(X1.reshape(N, 3))

def free_standing_bridge(g = 0.1, rho = 0, c = 10, k = 0.1, mu_1 = 1000, mu_2 = 0.1, N = 10,
                         tower_height = 4, tower_distances = 4, bridge_stretch = 1, max_iter = 500):

    consts = [g, rho, c, k]
    M = 0

    # Construct the two towers
    tower_one = np.array([  [0,0,0],
                            [0,2,0],
                            [2,0,0],
                            [2,2,0],
                            [0,1,tower_height],
                            [2,1,tower_height]])
    tower_two = np.copy(tower_one)
    tower_two[:,1] += tower_distances

    # Construct plank points
    left_cable_pts = np.array([[0, 1 + tower_distances/5  , tower_height + bridge_stretch],
                            [0, 1 + 2*tower_distances/5, tower_height + 2*bridge_stretch],
                            [0, 1 + 3*tower_distances/5, tower_height + 2*bridge_stretch],
                            [0, 1 + 4*tower_distances/5, tower_height + bridge_stretch]])
    right_cable_pts = np.copy(left_cable_pts)
    right_cable_pts[:,0] += 2

    # Assemble points
    X0 = np.vstack((tower_one, tower_two, left_cable_pts, right_cable_pts))
    N = np.size(X0,0)

    # Construct bars matrix
    bars = np.zeros((N, N))

    # Tower one
    bars[0,1] = 1
    bars[0,2] = 1
    bars[2,3] = 1
    bars[1,3] = 1
    bars[1,4] = 1
    bars[0,4] = 1
    bars[2,5] = 1
    bars[3,5] = 1
    bars[4,5] = 1

    # Tower two
    bars[0+6,1+6] = 1
    bars[0+6,2+6] = 1
    bars[2+6,3+6] = 1
    bars[1+6,3+6] = 1
    bars[1+6,4+6] = 1
    bars[0+6,4+6] = 1
    bars[2+6,5+6] = 1
    bars[3+6,5+6] = 1
    bars[4+6,5+6] = 1

    # Connect tower one and two at the base
    bars[3, 2+6] = 1
    bars[1, 0+6] = 1

    # Connect tower one and two at the top
    bars[4, 4+6] = 1
    bars[5, 5+6] = 1

    # Construct cables matrix
    cables = np.zeros((N, N))

    # Construct and connect First plank
    bars[0+12, 12+4] = 1
    cables[4, 12] = 1
    cables[5,12+4] = 1

    # Construct and connect 2.,3.,4. planks
    bars[1+12, 1+12+4] = 1
    cables[12,13] = 1
    cables[12+4,13+4] = 1

    bars[1+12+1, 1+12+4+1] = 1
    cables[12+1,13+1] = 1
    cables[12+4+1,13+4+1] = 1

    bars[1+12+2, 1+12+4+2] = 1
    cables[12+2,13+2] = 1
    cables[12+4+2,13+4+2] = 1

    # 4. plank to 2. tower connection
    cables[11,19] = 1
    cables[10, 15] = 1

    bars_indices = np.asarray(np.where(bars != 0))
    for i, j in bars_indices.T:
        bars[i,j] = np.linalg.norm(X0[i] - X0[j])

    cables_indices = np.asarray(np.where(cables != 0))
    for i, j in cables_indices.T:
        cables[i,j] = np.linalg.norm(X0[i] - X0[j]) * 0.95

    ms = np.ones(N) * 0.1
    #ms[0], ms[2], ms[1+6], ms[3+6] = [1] * 4

    def f(X):
        return efunc.Q(X, mu_1, mu_2, cables, bars, ms, consts)

    def df(X):
        return efunc.dQ(X, mu_1, mu_2, cables, bars, ms, consts, N)

    X0 = X0.flatten()
    X1, gradients = solver.BFGS(X0, N, 0, max_iter, f, df)
    cplot.plot_points(X0, X1, cables, bars, 0, gradients, title = 'Free standing bridge', plot_view = 'x', zprojection = False)
