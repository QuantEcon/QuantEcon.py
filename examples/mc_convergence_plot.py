"""
Filename: mc_convergence_plot.py
Authors: John Stachurski, Thomas J. Sargent

"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from quantecon import mc_compute_stationary

P = ((0.971, 0.029, 0.000),
     (0.145, 0.778, 0.077),
     (0.000, 0.508, 0.492))
P = np.array(P)

psi = (0.0, 0.2, 0.8)        # Initial condition

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_zlim(0, 1)
ax.set_xticks((0.25, 0.5, 0.75))
ax.set_yticks((0.25, 0.5, 0.75))
ax.set_zticks((0.25, 0.5, 0.75))

x_vals, y_vals, z_vals = [], [], []
for t in range(20):
    x_vals.append(psi[0])
    y_vals.append(psi[1])
    z_vals.append(psi[2])
    psi = np.dot(psi, P)

ax.scatter(x_vals, y_vals, z_vals, c='r', s=60)

psi_star = mc_compute_stationary(P)[0]
ax.scatter(psi_star[0], psi_star[1], psi_star[2], c='k', s=60)

plt.show()
