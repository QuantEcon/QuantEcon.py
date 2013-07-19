from compute_fp import compute_fixed_point
from matplotlib import pyplot as plt
from ifp import *

# === solve for optimal consumption === #
m = consumerProblem(r=0.03, grid_max=4)
v_init, c_init = initialize(m)
c = compute_fixed_point(coleman_operator, m, c_init)
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
fig.show()
