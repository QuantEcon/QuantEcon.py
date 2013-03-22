
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
import numpy as np
from odu_vfi import *
from compute_vf import compute_value_function


sp = searchProblem(w_grid_size=100, pi_grid_size=100)
v_init = np.zeros(len(sp.grid_points)) + sp.c / (1 - sp.beta)
v = compute_value_function(bellman, sp, v_init)
policy = get_greedy(sp, v)
# Make functions from these arrays by interpolation
vf = LinearNDInterpolator(sp.grid_points, v)
pf = LinearNDInterpolator(sp.grid_points, policy)

pi_plot_grid_size, w_plot_grid_size = 100, 100
pi_plot_grid = np.linspace(0.001, 0.99, pi_plot_grid_size)
w_plot_grid = np.linspace(0, sp.w_max, w_plot_grid_size)

#plot_choice = 'value_function'
plot_choice = 'policy_function'

if plot_choice == 'value_function':
    Z = np.empty((w_plot_grid_size, pi_plot_grid_size))
    for i in range(w_plot_grid_size):
        for j in range(pi_plot_grid_size):
            Z[i, j] = vf(w_plot_grid[i], pi_plot_grid[j])
    fig, ax = plt.subplots()
    ax.contourf(pi_plot_grid, w_plot_grid, Z, 12, alpha=0.6, cmap=cm.jet)
    cs = ax.contour(pi_plot_grid, w_plot_grid, Z, 12, colors="black")
    ax.clabel(cs, inline=1, fontsize=10)
    ax.set_xlabel('pi', fontsize=14)
    ax.set_ylabel('wage', fontsize=14)
    fig.show()
else:
    Z = np.empty((w_plot_grid_size, pi_plot_grid_size))
    for i in range(w_plot_grid_size):
        for j in range(pi_plot_grid_size):
            Z[i, j] = pf(w_plot_grid[i], pi_plot_grid[j])
    fig, ax = plt.subplots()
    ax.contourf(pi_plot_grid, w_plot_grid, Z, 1, alpha=0.6, cmap=cm.jet)
    ax.contour(pi_plot_grid, w_plot_grid, Z, 1, colors="black")
    ax.set_xlabel('pi', fontsize=14)
    ax.set_ylabel('wage', fontsize=14)
    ax.text(0.4, 1.0, 'reject')
    ax.text(0.7, 1.8, 'accept')
    fig.show()
