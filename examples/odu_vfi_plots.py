"""
Filename: odu_vfi_plots.py
Authors: John Stachurski and Thomas Sargent
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm
from scipy.interpolate import LinearNDInterpolator
import numpy as np
from quantecon import compute_fixed_point
from quantecon.models import SearchProblem


sp = SearchProblem(w_grid_size=100, pi_grid_size=100)
v_init = np.zeros(len(sp.grid_points)) + sp.c / (1 - sp.beta)
v = compute_fixed_point(sp.bellman_operator, v_init)
policy = sp.get_greedy(v)

# Make functions from these arrays by interpolation
vf = LinearNDInterpolator(sp.grid_points, v)
pf = LinearNDInterpolator(sp.grid_points, policy)

pi_plot_grid_size, w_plot_grid_size = 100, 100
pi_plot_grid = np.linspace(0.001, 0.99, pi_plot_grid_size)
w_plot_grid = np.linspace(0, sp.w_max, w_plot_grid_size)

# plot_choice = 'value_function'
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

plt.show()
