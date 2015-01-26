"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: lin_inter_3d_plot.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 21/08/2013
"""
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np

alpha = 0.7
phi_ext = 2 * 3.14 * 0.5


def f(a, b):
    # return 2 + alpha - 2 * np.cos(b)*np.cos(a) - alpha*np.cos(phi_ext - 2*b)
    return a + np.sqrt(b)

x_max = 3
y_max = 2.5

# === the approximation grid === #
Nx0, Ny0 = 25, 25
x0 = np.linspace(0, x_max, Nx0)
y0 = np.linspace(0, y_max, Ny0)
X0, Y0 = np.meshgrid(x0, y0)
points = np.column_stack((X0.ravel(1), Y0.ravel(1)))

# === generate the function values on the grid === #
Z0 = np.empty(Nx0 * Ny0)
for i in range(len(Z0)):
    a, b = points[i, :]
    Z0[i] = f(a, b)

g = LinearNDInterpolator(points, Z0)

# === a grid for plotting === #
Nx1, Ny1 = 100, 100
x1 = np.linspace(0, x_max, Nx1)
y1 = np.linspace(0, y_max, Ny1)
X1, Y1 = np.meshgrid(x1, y1)

# === the approximating function, as a matrix, for plotting === #
# ZA = np.empty((Ny1, Nx1))
# for i in range(Ny1):
#    for j in range(Nx1):
#        ZA[i, j] = g(x1[j], y1[i])
ZA = g(X1, Y1)
ZF = f(X1, Y1)

# === plot === #
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
p = ax.plot_wireframe(X1, Y1, ZF, rstride=4, cstride=4)
plt.show()
