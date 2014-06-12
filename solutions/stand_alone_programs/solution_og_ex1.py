import matplotlib.pyplot as plt
from quantecon.optgrowth import GrowthModel, bellman_operator, compute_greedy
from quantecon.compute_fp import compute_fixed_point

alpha, beta = 0.65, 0.95
gm = GrowthModel() 
true_sigma = (1 - alpha * beta) * gm.grid**alpha
w = 5 * gm.u(gm.grid) - 25  # Initial condition

fig, ax = plt.subplots(3, 1, figsize=(8, 10))

for i, n in enumerate((2, 4, 6)):
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(0, 2)
    ax[i].set_yticks((0, 1))
    ax[i].set_xticks((0, 2))

    v_star = compute_fixed_point(bellman_operator, gm, w, max_iter=n)
    sigma = compute_greedy(gm, v_star)

    ax[i].plot(gm.grid, sigma, 'b-', lw=2, alpha=0.8, label='approximate optimal policy')
    ax[i].plot(gm.grid, true_sigma, 'k-', lw=2, alpha=0.8, label='true optimal policy')
    ax[i].legend(loc='upper left')
    ax[i].set_title('{} value function iterations'.format(n))

plt.show()
