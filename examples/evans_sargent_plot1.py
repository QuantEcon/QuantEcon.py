"""
Plot 1 from the Evans Sargent model.

@author: David Evans
Edited by: John Stachurski

"""
import numpy as np
import matplotlib.pyplot as plt
from evans_sargent import T, y

tt = np.arange(T)  # tt is used to make the plot time index correct.

n_rows = 3
fig, axes = plt.subplots(n_rows, 1, figsize=(10, 12))

plt.subplots_adjust(hspace=0.5)
for ax in axes:
    ax.grid()
    ax.set_xlim(0, 15)

bbox = (0., 1.02, 1., .102)
legend_args = {'bbox_to_anchor': bbox, 'loc': 3, 'mode': 'expand'}
p_args = {'lw': 2, 'alpha': 0.7}

ax = axes[0]
ax.plot(tt, y[1, :], 'b-', label="output", **p_args)
ax.set_ylabel(r"$Q$", fontsize=16)
ax.legend(ncol=1, **legend_args)

ax = axes[1]
ax.plot(tt, y[2, :], 'b-', label="tax rate", **p_args)
ax.set_ylabel(r"$\tau$", fontsize=16)
ax.set_yticks((0.0, 0.2, 0.4, 0.6, 0.8))
ax.legend(ncol=1, **legend_args)

ax = axes[2]
ax.plot(tt, y[3, :], 'b-', label="first difference in output", **p_args)
ax.set_ylabel(r"$u$", fontsize=16)
ax.set_yticks((0, 100, 200, 300, 400))
ax.legend(ncol=1, **legend_args)
ax.set_xlabel(r'time', fontsize=16)

plt.show()
