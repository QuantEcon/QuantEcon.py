"""
Helps to illustrate the spectral density for AR(1) X' = phi X + epsilon
"""
import numpy as np
import matplotlib.pyplot as plt

phi = -0.8
times = range(16)
y1 = [phi**k / (1 - phi**2) for k in times]
y2 = [np.cos(np.pi * k) for k in times]
y3 = [a * b for a, b in zip(y1, y2)]

num_rows, num_cols = 3, 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
plt.subplots_adjust(hspace=0.25)

# Autocovariance when phi = -0.8
ax = axes[0]
ax.plot(times, y1, 'bo-', alpha=0.6, label=r'$\gamma(k)$')
ax.legend(loc='upper right')
ax.set_xlim(0, 15)
ax.set_yticks((-2, 0, 2))
ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

# Cycles at frequence pi
ax = axes[1]
ax.plot(times, y2, 'bo-', alpha=0.6, label=r'$\cos(\pi k)$')
ax.legend(loc='upper right')
ax.set_xlim(0, 15)
ax.set_yticks((-1, 0, 1))
ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

# Product
ax = axes[2]
ax.stem(times, y3, label=r'$\gamma(k) \cos(\pi k)$')
ax.legend(loc='upper right')
ax.set_xlim((0, 15))
ax.set_ylim(-3, 3)
ax.set_yticks((-1, 0, 1, 2, 3))
ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)

plt.show()
