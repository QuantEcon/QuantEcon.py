import matplotlib.pyplot as plt
from scipy.stats import binom

fig, axes = plt.subplots(2, 2)
plt.subplots_adjust(hspace=0.4)
axes = axes.flatten()
ns = [1, 2, 4, 8]
dom = list(range(9))

for ax, n in zip(axes, ns):
    b = binom(n, 0.5)
    ax.bar(dom, b.pmf(dom), alpha=0.6, align='center')
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(list(range(9)))
    ax.set_yticks((0, 0.2, 0.4))
    ax.set_title(r'$n = {}$'.format(n))

fig.show()
