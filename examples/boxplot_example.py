import numpy as np
import matplotlib.pyplot as plt

n = 500
x = np.random.randn(n)        # N(0, 1)
x = np.exp(x)                 # Map x to lognormal
y = np.random.randn(n) + 2.0  # N(2, 1)
z = np.random.randn(n) + 4.0  # N(4, 1)

fig, ax = plt.subplots(figsize=(10, 6.6))
ax.boxplot([x, y, z])
ax.set_xticks((1, 2, 3))
ax.set_ylim(-2, 14)
ax.set_xticklabels((r'$X$', r'$Y$', r'$Z$'), fontsize=16) 
plt.show()


