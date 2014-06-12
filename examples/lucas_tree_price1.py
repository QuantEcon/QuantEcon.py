
from __future__ import division  # Omit for Python 3.x
import numpy as np
import matplotlib.pyplot as plt
from quantecon.lucastree import lucas_tree, compute_lt_price

fig, ax = plt.subplots()
#grid = np.linspace(1e-10, 4, 100)

tree = lucas_tree(gamma=2, beta=0.95, alpha=0.90, sigma=0.1)
grid, price_vals = compute_lt_price(tree)
ax.plot(grid, price_vals, lw=2, alpha=0.7, label=r'$p^*(y)$')
ax.set_xlim(min(grid), max(grid))

#tree = lucas_tree(gamma=3, beta=0.95, alpha=0.90, sigma=0.1)
#grid, price_vals = compute_price(tree)
#ax.plot(grid, price_vals, lw=2, alpha=0.7, label='more patient')
#ax.set_xlim(min(grid), max(grid))

ax.set_xlabel(r'$y$', fontsize=16)
ax.set_ylabel(r'price', fontsize=16)
ax.legend(loc='upper left')

plt.show()

