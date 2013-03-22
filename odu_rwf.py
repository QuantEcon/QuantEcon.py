"""
Author: John Stachurski, with Thomas J. Sargent
Date:   3/2013
File:   odu_rwf.py

Solves the "Offer Distribution Unknown" Model by iterating on a guess of the
reservation wage function.
"""
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
from odu_vfi import searchProblem
from scipy.integrate import fixed_quad
from compute_fp import compute_fixed_point

def res_wage_operator(sp, w_bar):
    """
    Updates the reservation wage function guess via the operator Q.  Returns
    the updated function Q w_bar, represented as an array.
    
        * sp is an instance of searchProblem, defined in odu_vfi
        * w_bar is a NumPy array with len(w_bar) = len(sp.pi_grid)

    """
    beta, c, f, g, q = sp.beta, sp.c, sp.f, sp.g, sp.q    # Simplify names
    w_bar_f = lambda p: interp(p, sp.pi_grid, w_bar)   # Turn into function
    M = len(w_bar)
    new_w = np.empty(M)
    for i in range(M):
        pi = sp.pi_grid[i]
        integrand = lambda wp: np.maximum(wp, w_bar_f(q(wp, pi))) * \
                    (pi * f(wp) + (1 - pi) * g(wp))
        integral, error = fixed_quad(integrand, 0, sp.w_max)
        new_w[i] = (1 - beta) * c + beta * integral
    return new_w

if __name__ == '__main__':  # If module is run rather than imported
    sp = searchProblem(pi_grid_size=50)
    w_bar_init = np.ones(len(sp.pi_grid)) 
    w_bar = compute_fixed_point(res_wage_operator, sp, w_bar_init)
    fig, ax = plt.subplots()
    ax.plot(sp.pi_grid, w_bar, linewidth=2, color='black')
    ax.set_ylim(0, 2)
    ax.grid(axis='x', linewidth=0.25, linestyle='--', color='0.25')
    ax.grid(axis='y', linewidth=0.25, linestyle='--', color='0.25')
    ax.fill_between(sp.pi_grid, 0, w_bar, color='blue', alpha=0.15)
    ax.fill_between(sp.pi_grid, w_bar, 2, color='green', alpha=0.15)
    ax.text(0.42, 1.2, 'reject')
    ax.text(0.7, 1.8, 'accept')
    fig.show()
