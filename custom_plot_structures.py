import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
np.set_printoptions(suppress=True, precision=6)


def plot_points(X0, X1, bars, cables, M, gradients, keep_zlim = False, title = "Tensegrity system", simplify = False, plot_view = 'z'):
    """ Plotting 3D tensegrity system

    Args:
        X0 (array(3*N)): array of nodes in initial position
        X1 (array(3*N)): array of nodes after optimization
        bars (array(N, N)): neighbour array. bars[i, j] = l_ij. if bars[i, j] = 0, then there is no connetion
        cables (array(N, N)): neighbour array. cables[i, j] = l_ij. if cables[i, j] = 0, then there is no connetion
        M (int): number of fixed nodes
        title (str, optional): The title of the resulting 3D plot. Defaults to "Tensegrity system".
        simplify (bool, optional): Removes clutter in the plot to better show more complex systems. Defaults to False
        plot_view (char): Either 'x', 'y', or 'z'. Chooses which plane to plot. 
                          For example, 'z' would result in plotting 'xy' plane
    """
    fig = plt.figure(1, figsize = (20, 4))

    ax0 = fig.add_subplot(141, projection = '3d')   # Initial system
    ax1 = fig.add_subplot(142, projection = '3d')   # After optimization
    ax2 = fig.add_subplot(143)                      # Plot on a plane
    ax3 = fig.add_subplot(144)                      # Log-log of gradient

    ax0.set_title('Initial system')
    ax1.set_title(title)

    x0, y0, z0 = X0[0::3], X0[1::3], X0[2::3]
    x1, y1, z1 = X1[0::3], X1[1::3], X1[2::3]

    ax0.scatter(x0[:M], y0[:M], z0[:M], s = 50, c = 'red')     # Fixed nodes
    ax0.scatter(x0[M:], y0[M:], z0[M:], s = 50, c = 'blue')    # Variable nodes
    ax1.scatter(x1[:M], y1[:M], z1[:M], s = 50, c = 'red')     # Fixed nodes
    ax1.scatter(x1[M:], y1[M:], z1[M:], s = 50, c = 'blue')    # Variable nodes

    # Projection down on z
    if simplify == False:
        z0_lim = ax0.get_zlim3d()[0]
        z1_lim = ax1.get_zlim3d()[0]
        for i in range(len(z0)):
            ax0.plot([x0[i], x0[i]], [y0[i], y0[i]], [z0[i], z0_lim], linestyle = (0, (1, 5)), color = 'black')
            ax1.plot([x1[i], x1[i]], [y1[i], y1[i]], [z1[i], z1_lim], linestyle = (0, (1, 5)), color = 'black')

    for ax in [ax0, ax1]:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    if keep_zlim:
        ax1.set_zlim3d(ax0.get_zlim3d())

    # Plot on a plane
    if plot_view == 'z':                
        axis1_2d, axis2_2d = x1, y1
    elif plot_view == 'y':
        axis1_2d, axis2_2d = x1, z1
    elif plot_view == 'x':
        axis1_2d, axis2_2d = y1, z1
    else:
        return ValueError('plot_view needs to either be x, y or z')
    plane_string = 'xyz'.replace(plot_view, '')         # String manipulation for title and axis labels
    
    ax2.set_title(title + ', in ' + plane_string + ' plane' )
    ax2.scatter(axis1_2d[:M], axis2_2d[:M], s = 50, c = 'red')     # Fixed nodes
    ax2.scatter(axis1_2d[M:], axis2_2d[M:], s = 50, c = 'blue')    # Variable nodes
    ax2.grid(True)
    ax2.set_xlabel(plane_string[0])
    ax2.set_ylabel(plane_string[1])

    # Plotting cables and bars
    cable_indices = np.asarray(np.where(cables != 0))
    for i, j in cable_indices.T:
        ax0.plot([x0[i], x0[j]], [y0[i], y0[j]], [z0[i], z0[j]], '--', c = 'red')
        ax1.plot([x1[i], x1[j]], [y1[i], y1[j]], [z1[i], z1[j]], '--', c = 'red')
        ax2.plot([axis1_2d[i], axis1_2d[j]], [axis2_2d[i], axis2_2d[j]], '--', c = 'red')

    bar_indices = np.asarray(np.where(bars != 0))
    for i, j in bar_indices.T:
        ax0.plot([x0[i], x0[j]], [y0[i], y0[j]], [z0[i], z0[j]], '-', c = 'blue')
        ax1.plot([x1[i], x1[j]], [y1[i], y1[j]], [z1[i], z1[j]], '-', c = 'blue')
        ax2.plot([axis1_2d[i], axis1_2d[j]], [axis2_2d[i], axis2_2d[j]], '-', c = 'blue')

    # Plotting norm of gradient in each iteration
    ax3.set_title(r'Log-log plot of $||\nabla E||_2$ in each iteration')
    ax3.loglog(gradients)
    ax3.grid('True')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel(r'$||\nabla E||_2$')

    for ax in [ax0, ax1, ax2]:                  # Fixing tick sizes
        ax.locator_params(axis='both', nbins=5)
        ax.tick_params(axis='both', labelsize=11)
    plt.rc('axes', labelsize=12)                # Label sizes


    plt.subplots_adjust(wspace = 0.4)           # Adding 
    plt.show()

