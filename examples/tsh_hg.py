
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from quantecon import LinearStateSpace

phi_1, phi_2, phi_3, phi_4 = 0.5, -0.2, 0, 0.5
sigma = 0.1

A = [[phi_1, phi_2, phi_3, phi_4],
     [1,     0,     0,     0],
     [0,     1,     0,     0],
     [0,     0,     1,     0]]
C = [sigma, 0, 0, 0]
G = [1, 0, 0, 0]

T = 30
ar = LinearStateSpace(A, C, G)

ymin, ymax = -0.8, 1.25

fig, ax = plt.subplots(figsize=(8, 4))

ax.set_xlim(ymin, ymax)
ax.set_xlabel(r'$y_t$', fontsize=16)

x, y = ar.replicate(T=T, num_reps=100000)
mu_x, mu_y, Sigma_x, Sigma_y = ar.stationary_distributions()
f_y = norm(loc=float(mu_y), scale=float(np.sqrt(Sigma_y)))

y = y.flatten()
ax.hist(y, bins=50, normed=True, alpha=0.4)

ygrid = np.linspace(ymin, ymax, 150)
ax.plot(ygrid, f_y.pdf(ygrid), 'k-', lw=2, alpha=0.8, label='true density')
ax.legend()
plt.show()
