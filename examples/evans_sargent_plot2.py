"""
Plot 2 from the Evans Sargent model.

@author: David Evans
Edited by: John Stachurski

"""
import numpy as np
import matplotlib.pyplot as plt
from evans_sargent import T, uhatdif, tauhatdif, mu, G

tt = np.arange(T)  # tt is used to make the plot time index correct.
tt2 = np.arange(T-1)

n_rows = 4
fig, axes = plt.subplots(n_rows, 1, figsize=(10, 16))

plt.subplots_adjust(hspace=0.5)
for ax in axes:
    ax.grid(alpha=.5)
    ax.set_xlim(-0.5, 15)

bbox = (0., 1.02, 1., .102)
legend_args = {'bbox_to_anchor': bbox, 'loc': 3, 'mode': 'expand'}
p_args = {'lw': 2, 'alpha': 0.7}

ax = axes[0]
ax.plot(tt2, tauhatdif, label=r'time inconsistency differential for tax rate',
        **p_args)
ax.set_ylabel(r"$\Delta\tau$", fontsize=16)
ax.set_ylim(-0.1, 1.4)
ax.set_yticks((0.0, 0.4, 0.8, 1.2))
ax.legend(ncol=1, **legend_args)

ax = axes[1]
ax.plot(tt, uhatdif, label=r'time inconsistency differential for $u$',
        **p_args)
ax.set_ylabel(r"$\Delta u$", fontsize=16)
ax.set_ylim(-3, .1)
ax.set_yticks((-3.0, -2.0, -1.0, 0.0))
ax.legend(ncol=1, **legend_args)

ax = axes[2]
ax.plot(tt, mu, label='Lagrange multiplier', **p_args)
ax.set_ylabel(r"$\mu$", fontsize=16)
ax.set_ylim(2.34e-3, 2.52e-3)
ax.set_yticks((2.34e-3, 2.43e-3, 2.52e-3))
ax.legend(ncol=1, **legend_args)

ax = axes[3]
ax.plot(tt, G, label='government revenue', **p_args)
ax.set_ylabel(r"$G$", fontsize=16)
ax.set_ylim(9100, 9800)
ax.set_yticks((9200, 9400, 9600, 9800))
ax.legend(ncol=1, **legend_args)

ax.set_xlabel(r'time', fontsize=16)

plt.show()
# lines = plt.plot(tt, GPay, "o")
