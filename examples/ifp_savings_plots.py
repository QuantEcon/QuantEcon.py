"""
Origin: QE by John Stachurski and Thomas J. Sargent
Filename: ifp_savings_plots.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

"""
from compute_fp import compute_fixed_point
from matplotlib import pyplot as plt
from quantecon.compute_fp import compute_fixed_point
import quantecon as qe

# === solve for optimal consumption === #
m = qe.ifp.consumerProblem(r=0.03, grid_max=4)
v_init, c_init = qe.ifp.initialize(m)
c = compute_fixed_point(qe.ifp.coleman_operator, m, c_init)
a = m.asset_grid
R, z_vals = m.R, m.z_vals

# === generate savings plot === #
fig, ax = plt.subplots()
ax.plot(a, R * a + z_vals[0] - c[:, 0], label='low income')
ax.plot(a, R * a + z_vals[1] - c[:, 1], label='high income')
ax.plot(a, a, 'k--')
ax.set_xlabel('current assets')
ax.set_ylabel('next period assets')
ax.legend(loc='upper left')
plt.show()
