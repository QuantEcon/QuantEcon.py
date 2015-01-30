
from __future__ import division  # Omit for Python 3.x
import matplotlib.pyplot as plt
from quantecon.models import LucasTree

fig, ax = plt.subplots()

tree = LucasTree(gamma=2, beta=0.95, alpha=0.90, sigma=0.1)
grid, price_vals = tree.grid, tree.compute_lt_price()
ax.plot(grid, price_vals, lw=2, alpha=0.7, label=r'$p^*(y)$')
ax.set_xlim(min(grid), max(grid))

# tree = LucasTree(gamma=3, beta=0.95, alpha=0.90, sigma=0.1)
# grid, price_vals = tree.grid, tree.compute_lt_price()
# ax.plot(grid, price_vals, lw=2, alpha=0.7, label='more patient')
# ax.set_xlim(min(grid), max(grid))

ax.set_xlabel(r'$y$', fontsize=16)
ax.set_ylabel(r'price', fontsize=16)
ax.legend(loc='upper left')

plt.show()
