from compute_fp import compute_fixed_point
from matplotlib import pyplot as plt
import numpy as np
from ifp import coleman_operator, consumerProblem, initialize

r_vals = np.linspace(0, 0.04, 4)  

fig, ax = plt.subplots()
for r_val in r_vals:
    m = consumerProblem(r=r_val)
    v_init, c_init = initialize(m)
    c = compute_fixed_point(coleman_operator, m, c_init)
    ax.plot(m.asset_grid, c[:, 0], label=r'$r = %.3f$' % r_val)

ax.set_xlabel('asset level')
ax.set_ylabel('consumption (low income)')
ax.legend(loc='upper left')
fig.show()

