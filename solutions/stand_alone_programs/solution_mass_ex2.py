from __future__ import division  # Omit for Python 3.x
import numpy as np
import matplotlib.pyplot as plt
from quantecon import lucas_tree, compute_lt_price

fig, ax = plt.subplots()

ax.set_xlabel(r'$y$', fontsize=16)
ax.set_ylabel(r'price', fontsize=16)

for beta in (.95, 0.98):
    tree = lucas_tree(gamma=2, beta=beta, alpha=0.90, sigma=0.1)
    grid, price_vals = compute_lt_price(tree)
    label = r'$\beta = {}$'.format(beta)
    ax.plot(grid, price_vals, lw=2, alpha=0.7, label=label)

ax.legend(loc='upper left')
ax.set_xlim(min(grid), max(grid))
plt.show()
