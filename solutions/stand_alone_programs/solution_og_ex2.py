import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from quantecon import GrowthModel
from quantecon import compute_fixed_point

gm = GrowthModel() 
w = 5 * gm.u(gm.grid) - 25  # To be used as an initial condition
discount_factors = (0.9, 0.94, 0.98)
series_length = 25

fig, ax = plt.subplots()
ax.set_xlabel("time")
ax.set_ylabel("capital")

for beta in discount_factors:

    # Compute the optimal policy given the discount factor
    gm.beta = beta
    v_star = compute_fixed_point(gm.bellman_operator, w, max_iter=20)
    sigma = gm.compute_greedy(v_star)

    # Compute the corresponding time series for capital
    k = np.empty(series_length)
    k[0] = 0.1
    sigma_function = lambda x: interp(x, gm.grid, sigma)
    for t in range(1, series_length):
        k[t] = gm.f(k[t-1]) - sigma_function(k[t-1])
    ax.plot(k, 'o-', lw=2, alpha=0.75, label=r'$\beta = {}$'.format(beta))

ax.legend(loc='lower right')
plt.show()


