from matplotlib import pyplot as plt
import numpy as np
from quantecon import compute_fixed_point, ConsumerProblem  
from solution_ifp_ex3 import compute_asset_series

M = 25
r_vals = np.linspace(0, 0.04, M)  
fig, ax = plt.subplots()

for b in (1, 3):
    asset_mean = []
    for r_val in r_vals:
        cp = ConsumerProblem(r=r_val, b=b)
        mean = np.mean(compute_asset_series(cp, T=250000))
        asset_mean.append(mean)
    ax.plot(asset_mean, r_vals, label=r'$b = %d$' % b)

ax.set_yticks(np.arange(.0, 0.045, .01))
ax.set_xticks(np.arange(-3, 2, 1))
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.grid(True)
ax.legend(loc='upper left')
plt.show()

