import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)


def plot_points(X0, X1, bars, cables, M, gradients, keep_zlim = False, title = "Tensegrity system", simplify = False):
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
    fig = plt.figure(1, figsize = (20, 4))

    ax0 = fig.add_subplot(141, projection = '3d')
    ax1 = fig.add_subplot(142, projection = '3d')
    ax2 = fig.add_subplot(143)
    ax3 = fig.add_subplot(144)

    ax0.set_title('Initial system')
    ax1.set_title(title)

    x0, y0, z0 = X0[0::3], X0[1::3], X0[2::3]
    x1, y1, z1 = X1[0::3], X1[1::3], X1[2::3]
    
    xlims, ylims, zlims = [None, None, None]
    if xlims == None:
        xmin, xmax = np.min(x0), np.max(x0)
        xlims = [xmin - (xmax-xmin) / 4, xmax + (xmax-xmin) / 4]
    if ylims == None:
        ymin, ymax = np.min(y0), np.max(y0)
        ylims = [ymin - (ymax-ymin) / 4, ymax + (ymax-ymin) / 4]
    if zlims == None:
        zmin, zmax = np.min(z0), np.max(z0)
        zlims = [zmin - (zmax-zmin) / 4, zmax + (zmax-zmin) / 4]

    ax0.scatter(x0[:M], y0[:M], z0[:M], s = 50, c = 'red')     # Fixed nodes
    ax0.scatter(x0[M:], y0[M:], z0[M:], s = 50, c = 'blue')    # Variable nodes
    ax1.scatter(x1[:M], y1[:M], z1[:M], s = 50, c = 'red')     # Fixed nodes
    ax1.scatter(x1[M:], y1[M:], z1[M:], s = 50, c = 'blue')    # Variable nodes

    # Projection down on z
    z0_lim = ax0.get_zlim3d()[0]
    z1_lim = ax1.get_zlim3d()[0]
    if simplify == False:
        for i in range(len(z0)):
            ax0.plot([x0[i], x0[i]], [y0[i], y0[i]], [z0[i], z0_lim], linestyle = (0, (1, 5)), color = 'black')
            ax1.plot([x1[i], x1[i]], [y1[i], y1[i]], [z1[i], z1_lim], linestyle = (0, (1, 5)), color = 'black')

    ax0.set_xlabel('x')
    ax0.set_ylabel('y')
    ax0.set_zlabel('z')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')

    # Top-down view
    ax2.set_title(title + ', top-down view')
    ax2.scatter(x1[:M], y1[:M], s = 50, c = 'red')     # Fixed nodes
    ax2.scatter(x1[M:], y1[M:], s = 50, c = 'blue')    # Variable nodes
    ax2.grid(True)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')

    cable_indices = np.asarray(np.where(cables != 0))
    for i, j in cable_indices.T:
        ax0.plot([x0[i], x0[j]], [y0[i], y0[j]], [z0[i], z0[j]], '--', c = 'red')
        ax1.plot([x1[i], x1[j]], [y1[i], y1[j]], [z1[i], z1[j]], '--', c = 'red')
        ax2.plot([x1[i], x1[j]], [y1[i], y1[j]], '--', c = 'red')

    bar_indices = np.asarray(np.where(bars != 0))
    for i, j in bar_indices.T:
        ax0.plot([x0[i], x0[j]], [y0[i], y0[j]], [z0[i], z0[j]], '-', c = 'blue')
        ax1.plot([x1[i], x1[j]], [y1[i], y1[j]], [z1[i], z1[j]], '-', c = 'blue')
        ax2.plot([x1[i], x1[j]], [y1[i], y1[j]], '-', c = 'blue')

    # Norm of gradient in each iteration
    ax3.set_title(r'Log-log plot of $||\nabla E||_2$ in each iteration')
    ax3.loglog(gradients)
    ax3.grid('True')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel(r'$||\nabla E||_2$')
    
    plt.show()

