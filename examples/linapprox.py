import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def f(x):
    y1 = 2 * np.cos(6 * x) + np.sin(14 * x)
    return y1 + 2.5

c_grid = np.linspace(0, 1, 6)


def Af(x):
    return sp.interp(x, c_grid, f(c_grid))

f_grid = np.linspace(0, 1, 150)

fig, ax = plt.subplots()
ax.set_xlim(0, 1)

ax.plot(f_grid, f(f_grid), 'b-', lw=2, alpha=0.8, label='true function')
ax.plot(f_grid, Af(f_grid), 'g-', lw=2, alpha=0.8,
        label='linear approximation')

ax.vlines(c_grid, c_grid * 0, f(c_grid), linestyle='dashed', alpha=0.5)
ax.legend(loc='upper center')

plt.show()
