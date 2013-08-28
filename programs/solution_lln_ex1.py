"""
Illustrates the delta method, a consequence of the central limit theorem.
"""
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from matplotlib import rc

# == Specifying font, needs LaTeX integration == #
rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# == Set parameters == #
n = 250
replications = 100000
distribution = uniform(loc=0, scale=(np.pi / 2))
mu, s = distribution.mean(), distribution.std()

g = np.sin
g_prime = np.cos

# == Generate obs of sqrt{n} (g(\bar X_n) - g(\mu)) == #
data = distribution.rvs((replications, n)) 
sample_means = data.mean(axis=1)  # Compute mean of each row
error_obs = np.sqrt(n) * (g(sample_means) - g(mu))

# == Plot == #
asymptotic_sd = g_prime(mu) * s
fig, ax = plt.subplots()
xmin = -3 * g_prime(mu) * s
xmax = -xmin
ax.set_xlim(xmin, xmax)
ax.hist(error_obs, bins=60, alpha=0.5, normed=True)
xgrid = np.linspace(xmin, xmax, 200)
lb = r"$N(0, g'(\mu)^2  \sigma^2)$"
ax.plot(xgrid, norm.pdf(xgrid, scale=asymptotic_sd), 'k-', lw=2, label=lb)
ax.legend()

fig.show()



