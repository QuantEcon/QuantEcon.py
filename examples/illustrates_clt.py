"""
Filename: illustrates_clt.py
Authors: John Stachurski and Thomas J. Sargent

Visual illustration of the central limit theorem.  Histograms draws of

    Y_n := \sqrt{n} (\bar X_n - \mu)

for a given distribution of X_i, and a given choice of n.
"""
import numpy as np
from scipy.stats import expon, norm
import matplotlib.pyplot as plt
from matplotlib import rc

# == Specifying font, needs LaTeX integration == #
rc('font', **{'family': 'serif', 'serif': ['Palatino']})
rc('text', usetex=True)

# == Set parameters == #
n = 250     # Choice of n
k = 100000  # Number of draws of Y_n
distribution = expon(2)  # Exponential distribution, lambda = 1/2
mu, s = distribution.mean(), distribution.std()

# == Draw underlying RVs. Each row contains a draw of X_1,..,X_n == #
data = distribution.rvs((k, n))
# == Compute mean of each row, producing k draws of \bar X_n == #
sample_means = data.mean(axis=1)
# == Generate observations of Y_n == #
Y = np.sqrt(n) * (sample_means - mu)

# == Plot == #
fig, ax = plt.subplots()
xmin, xmax = -3 * s, 3 * s
ax.set_xlim(xmin, xmax)
ax.hist(Y, bins=60, alpha=0.5, normed=True)
xgrid = np.linspace(xmin, xmax, 200)
ax.plot(xgrid, norm.pdf(xgrid, scale=s), 'k-', lw=2, label=r'$N(0, \sigma^2)$')
ax.legend()

plt.show()
