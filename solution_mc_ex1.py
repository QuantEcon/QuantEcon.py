
import numpy as np
import matplotlib.pyplot as plt
import mc_tools

alpha = beta = 0.1
N = 10000
p = beta / (alpha + beta)

P = ((1 - alpha, alpha),   # Careful: P and p are distinct
     (beta, 1 - beta))
P = np.array(P)

fig, ax = plt.subplots()
ax.set_ylim(-0.25, 0.25)
ax.grid()
ax.hlines(0, 0, N, lw=2, alpha=0.6)  # Horizonal line at zero

for x0, col in ((0, 'blue'), (1, 'green')):
    X = mc_tools.sample_path(P, x0, N)
    X_bar = X.cumsum() / (1 + np.arange(N, dtype=float)) 
    ax.fill_between(range(N), np.zeros(N), X_bar - p, color=col, alpha=0.1)
    ax.plot(X_bar - p, color=col, label=r'$X_0 = \, {} $'.format(x0))
    ax.plot(X_bar - p, 'k-', alpha=0.6)  # Overlay in black--make lines clearer

ax.legend(loc='upper right')
fig.show()
