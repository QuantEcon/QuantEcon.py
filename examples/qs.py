
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from matplotlib import cm

xmin, xmax = -4, 12
x = 10
alpha = 0.5

m, v = x, 10

xgrid = np.linspace(xmin, xmax, 200)

fig, ax = plt.subplots()

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['left'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))

ax.set_ylim(-0.05, 0.5)
ax.set_xticks((x,))
ax.set_xticklabels((r'$x$',), fontsize=18)
ax.set_yticks(())

K = 3
for i in range(K):
    m = alpha * m
    v = alpha * alpha * v + 1
    f = norm(loc=m, scale=np.sqrt(v))
    k = (i + 0.5) / K
    ax.plot(xgrid, f.pdf(xgrid), lw=1, color='black', alpha=0.4)
    ax.fill_between(xgrid, 0 * xgrid, f.pdf(xgrid), color=cm.jet(k), alpha=0.4)


ax.annotate(r'$Q(x,\cdot)$', xy=(6.6, 0.2),  xycoords='data',
            xytext=(20, 90), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))
ax.annotate(r'$Q^2(x,\cdot)$', xy=(3.6, 0.24),  xycoords='data',
            xytext=(20, 90), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2"))
ax.annotate(r'$Q^3(x,\cdot)$', xy=(-0.2, 0.28),  xycoords='data',
            xytext=(-90, 90), textcoords='offset points', fontsize=16,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
fig.show()
