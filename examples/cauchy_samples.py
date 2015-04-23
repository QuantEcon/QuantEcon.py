
import numpy as np
from scipy.stats import cauchy
import matplotlib.pyplot as plt

n = 1000
distribution = cauchy()

fig, ax = plt.subplots()
data = distribution.rvs(n)

if 0:
    ax.plot(list(range(n)), data, 'bo', alpha=0.5)
    ax.vlines(list(range(n)), 0, data, lw=0.2)
    ax.set_title("{} observations from the Cauchy distribution".format(n))

if 1:
    # == Compute sample mean at each n == #
    sample_mean = np.empty(n)
    for i in range(n):
        sample_mean[i] = np.mean(data[:i])

    # == Plot == #
    ax.plot(list(range(n)), sample_mean, 'r-', lw=3, alpha=0.6,
            label=r'$\bar X_n$')
    ax.plot(list(range(n)), [0] * n, 'k--', lw=0.5)
    ax.legend()

fig.show()
