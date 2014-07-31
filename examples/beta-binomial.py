"""
Filename: beta-binomial.py
Authors: John Stachurski, Thomas J. Sargent

"""
import matplotlib.pyplot as plt
from quantecon.distributions import BetaBinomial

n = 50
a_vals = [0.5, 1, 100]
b_vals = [0.5, 1, 100]
fig, ax = plt.subplots()
for a, b in zip(a_vals, b_vals):
    ab_label = r"$a = %.1f$, $b = %.1f$" % (a, b)
    ax.plot(list(range(0, n+1)), BetaBinomial(n, a, b).pdf(), "-o",
            label=ab_label)
ax.legend()
plt.show()
