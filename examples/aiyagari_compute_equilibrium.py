
import numpy as np
import quantecon as qe
import matplotlib.pyplot as plt
from numba import jit
from aiyagari_household import Household, asset_marginal
from quantecon.markov import DiscreteDP


A = 2.5
N = 0.05
alpha = 0.33
beta = 0.96


def r_to_w(r):
    return A * (1 - alpha) * (alpha / (1 + r))**(alpha / (1 - alpha))

def rd(K):
    return A * alpha * (N / K)**(1 - alpha)


def prices_to_capital_stock(am, r):
    """
    Map prices to the induced level of capital stock.
    
    Paramters:
    ----------
    
    am : AiyagariModel
        An instance of an Aiyagari economy
    r : float
        The interest rate
    """
    w = r_to_w(r)
    am.set_prices(r, w)
    aiyagari_ddp = DiscreteDP(am.R, am.Q, beta)
    # Compute the optimal policy
    results = aiyagari_ddp.solve(method='policy_iteration')
    # Compute the stationary distribution
    stationary_probs = results.mc.stationary_distributions[0]
    # Extract the marginal distribution for assets
    asset_probs = asset_marginal(stationary_probs, am.a_size, am.z_size)
    # Return K
    return np.sum(asset_probs * am.a_vals)  


# Create an instance of Household 
am = Household(a_max=20)

# Use the instance to build a discrete dynamic program
am_ddp = DiscreteDP(am.R, am.Q, am.beta)

# Create a grid of r values at which to compute demand and supply of capital
num_points = 20
r_vals = np.linspace(0.0, 0.04, num_points)

# Compute supply of capital
k_vals = np.empty(num_points)
for i, r in enumerate(r_vals):
    k_vals[i] = prices_to_capital_stock(am, r)

# Plot against demand for capital by firms
fig, ax = plt.subplots(figsize=(11, 8))
ax.plot(k_vals, r_vals, lw=2, alpha=0.6, label='supply of capital')
ax.plot(k_vals, rd(k_vals), lw=2, alpha=0.6, label='demand for capital')
ax.grid()
ax.set_xlabel('capital')
ax.set_ylabel('interest rate')
ax.legend(loc='upper right')

plt.show()
