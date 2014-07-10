import numpy as np
import matplotlib.pyplot as plt
x = np.random.randn(100)        # N(0, 1)
y = np.random.randn(100) + 2.0  # N(2, 1)
z = np.random.randn(100) + 4.0  # N(4, 1)
x = np.exp(x)  # Turn x into a lognormal
fig, ax = plt.subplots()
ax.boxplot([x, y, z])
plt.show()


