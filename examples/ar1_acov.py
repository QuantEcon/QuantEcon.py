"""
Plots autocovariance function for AR(1) X' = phi X + epsilon
"""
import numpy as np
import matplotlib.pyplot as plt
num_rows, num_cols = 2, 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)
# Autocovariance when phi = 0.8
temp = r'autocovariance, $\phi = {0:.2}$'
for i, phi in enumerate((0.8, -0.8)):
    ax = axes[i]
    times = range(16)
    acov = [phi**k / (1 - phi**2) for k in times]
    ax.plot(times, acov, 'bo-', alpha=0.6, label=temp.format(phi))
    ax.legend(loc='upper right')
    ax.set_xlabel('time')
    ax.set_xlim((0, 15))
    ax.hlines(0, 0, 15, linestyle='--', alpha=0.5)
plt.show()
