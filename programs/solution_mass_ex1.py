"""
Filename: solution_mass_ex1.py
Authors: David Evans, John Stachurski and Thomas J. Sargent
LastModified: 12/02/2014
"""

import numpy as np
import asset_pricing 

# == Define primitives == #
n = 5
P = 0.0125 * np.ones((n, n))
P += np.diag(0.95 - 0.0125 * np.ones(5))
lamb = np.array([1.05, 1.025, 1.0, 0.975, 0.95])
gamma = 2.0
beta = 0.94
zeta = 1.0

ap = asset_pricing.AssetPrices(beta, P, lamb, gamma)

v = ap.tree_price()
print "Lucas Tree Prices: ", v

v_consol = ap.consol_price(zeta)
print "Consol Bond Prices: ", v_consol

P_tilde = P*lamb**(1-gamma)
temp = beta * P_tilde.dot(v) - beta * P_tilde.dot(np.ones(n))
print "Should be 0: ",  v - temp 

p_s = 150.0
w_bar, w_bars = ap.call_option(zeta, p_s, T = [10,20,30])
