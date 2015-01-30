"""
Filename: beta-binomial.py
Authors: John Stachurski, Thomas J. Sargent

"""
from scipy.special import binom, beta
import matplotlib.pyplot as plt
import numpy as np


def gen_probs(n, a, b):
    probs = np.zeros(n+1)
    for k in range(n+1):
        probs[k] = binom(n, k) * beta(k + a, n - k + b) / beta(a, b)
    return probs

n = 50
a_vals = [0.5, 1, 100]
b_vals = [0.5, 1, 100]
fig, ax = plt.subplots()
for a, b in zip(a_vals, b_vals):
    ab_label = r'$a = %.1f$, $b = %.1f$' % (a, b)
    ax.plot(list(range(0, n+1)), gen_probs(n, a, b), '-o', label=ab_label)
ax.legend()
plt.show()
