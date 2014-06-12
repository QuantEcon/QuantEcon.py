import matplotlib.pyplot as plt
import random
from quantecon import JvWorker, compute_fixed_point
import numpy as np

# Set up
wp = JvWorker(grid_size=25)
G, pi, F = wp.G, wp.pi, wp.F       # Simplify names

v_init = wp.x_grid * 0.5
V = compute_fixed_point(wp.bellman_operator, v_init, max_iter=40)
s_policy, phi_policy = wp.bellman_operator(V, return_policies=True)

# Turn the policy function arrays into actual functions
s = lambda y: np.interp(y, wp.x_grid, s_policy)
phi = lambda y: np.interp(y, wp.x_grid, phi_policy)

def h(x, b, U):
    return (1 - b) * G(x, phi(x)) + b * max(G(x, phi(x)), U)

plot_grid_max, plot_grid_size = 1.2, 100
plot_grid = np.linspace(0, plot_grid_max, plot_grid_size)
fig, ax = plt.subplots()
ax.set_xlim(0, plot_grid_max)
ax.set_ylim(0, plot_grid_max)
ticks = (0.25, 0.5, 0.75, 1.0)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel(r'$x_t$', fontsize=16)
ax.set_ylabel(r'$x_{t+1}$', fontsize=16, rotation='horizontal')

ax.plot(plot_grid, plot_grid, 'k--')  # 45 degree line
for x in plot_grid:
    for i in range(50):
        b = 1 if random.uniform(0, 1) < pi(s(x)) else 0
        U = wp.F.rvs(1)
        y = h(x, b, U)
        ax.plot(x, y, 'go', alpha=0.25)
plt.show()
