"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: jv_test.py
Authors: John Stachurski and Thomas Sargent
LastModified: 11/08/2013

Tests jv.py with a particular parameterization.

"""
import matplotlib.pyplot as plt
from jv import workerProblem, bellman_operator
from compute_fp import compute_fixed_point

# === solve for optimal policy === #
wp = workerProblem(grid_size=25)
v_init = wp.x_grid * 0.5
V = compute_fixed_point(bellman_operator, wp, v_init, max_iter=40)
s_policy, phi_policy = bellman_operator(wp, V, return_policies=True)

# === plot policies === #
fig, ax = plt.subplots()
ax.set_xlim(0, max(wp.x_grid))
ax.set_ylim(-0.1, 1.1)
ax.plot(wp.x_grid, phi_policy, 'b-', label='phi')
ax.plot(wp.x_grid, s_policy, 'g-', label='s')
ax.legend()
plt.show()

