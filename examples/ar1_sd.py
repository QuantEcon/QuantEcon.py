"""
Plots spectral density for AR(1) X' = phi X + epsilon
"""
import numpy as np
import matplotlib.pyplot as plt


def ar1_sd(phi, omega):
    return 1 / (1 - 2 * phi * np.cos(omega) + phi**2)

omegas = np.linspace(0, np.pi, 180)
num_rows, num_cols = 2, 1
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 8))
plt.subplots_adjust(hspace=0.4)

# Autocovariance when phi = 0.8
temp = r'spectral density, $\phi = {0:.2}$'
for i, phi in enumerate((0.8, -0.8)):
    ax = axes[i]
    sd = ar1_sd(phi, omegas)
    ax.plot(omegas, sd, 'b-', alpha=0.6, lw=2, label=temp.format(phi))
    ax.legend(loc='upper center')
    ax.set_xlabel('frequency')
    ax.set_xlim((0, np.pi))
plt.show()
