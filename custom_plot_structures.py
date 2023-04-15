import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)


def plot_points(X, bars, cables, M, lims = [None, None, None], title = "Tensegrity system", simplify = False):
    """ Plotting 3D tensegrity system

    Args:
        X (array(3*N)): array of nodes
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        M (int): number of fixed nodes
        lims (list, optional): . Defaults to [None, None, None].
        title (str, optional): The title of the plot. Defaults to "Tensegrity system".
        simplify (bool, optional): Removes clutter in the plot to better show more complex systems. Defaults to False

    Returns:
        list of limits: limits in x, y, and z axis
    """
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(projection = '3d')
    ax.set_title(title)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    
    xlims, ylims, zlims = lims
    if xlims == None:
        xmin, xmax = np.min(X[:,0]), np.max(X[:,0])
        xlims = [xmin - (xmax-xmin) / 4, xmax + (xmax-xmin) / 4]
    if ylims == None:
        ymin, ymax = np.min(X[:,1]), np.max(X[:,1])
        ylims = [ymin - (ymax-ymin) / 4, ymax + (ymax-ymin) / 4]
    if zlims == None:
        zmin, zmax = np.min(X[:,2]), np.max(X[:,2])
        zlims = [zmin - (zmax-zmin) / 4, zmax + (zmax-zmin) / 4]

    ax.scatter(x[:M], y[:M], z[:M], s = 100, c = 'red')
    ax.scatter(x[M:], y[M:], z[M:], s = 100, c = 'blue')

    if simplify == False:
        ax.plot(x, z, 'r+', zdir='y', zs = ylims[1])    # Projection on xz plane
        ax.plot(y, z, 'g+', zdir='x', zs = xlims[0])    # Projection on yz plane

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.set_xlim3d(xlims[0], xlims[1])
    ax.set_ylim3d(ylims[0], ylims[1])
    ax.set_zlim3d(zlims[0], zlims[1])

    cable_indices = np.asarray(np.where(cables != 0))
    for i, j in cable_indices.T:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b--')

    bar_indices = np.asarray(np.where(bars != 0))
    for i, j in bar_indices.T:
        ax.plot([x[i], x[j]], [y[i], y[j]], [z[i], z[j]], 'b-')

    # Projection down on z
    if simplify == False:
        for i in range(len(z)):
            ax.plot([x[i], x[i]], [y[i], y[i]], [z[i], zlims[0]], linestyle = (0, (1, 5)), color = 'black')
    plt.show() 
    return [xlims, ylims, zlims]