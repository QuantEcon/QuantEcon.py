

import numpy as np

alpha = 0.7
phi_ext = 2 * 3.14 * 0.5

def f(a, b):
    return 2 + alpha - 2 * np.cos(b)*np.cos(a) - alpha * np.cos(phi_ext - 2*b)
    #return a + np.sqrt(b)

x_max = 3
y_max = 2.5

# A grid for plotting
Nx1, Ny1 = 10, 5
x1 = np.linspace(0, x_max, Nx1)
y1 = np.linspace(0, y_max, Ny1)
X1, Y1 = np.meshgrid(x1, y1)

Zm = f(X1, Y1)
ZA = np.empty((Ny1, Nx1))
for i in range(Ny1):
    for j in range(Nx1):
        ZA[i, j] = f(x1[j], y1[i])


