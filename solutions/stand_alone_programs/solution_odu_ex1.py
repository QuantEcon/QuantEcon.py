"""
Solves the "Offer Distribution Unknown" model by iterating on a guess of the
reservation wage function.
"""
from scipy import interp
import numpy as np
from numpy import maximum as npmax
import matplotlib.pyplot as plt
from odu_vfi import searchProblem
from scipy.integrate import fixed_quad
from compute_fp import compute_fixed_point


def res_wage_operator(sp, phi):
    """
    Updates the reservation wage function guess phi via the operator Q.
    Returns the updated function Q phi, represented as the array new_phi.
    
        * sp is an instance of searchProblem, defined in odu_vfi
        * phi is a NumPy array with len(phi) = len(sp.pi_grid)

    """
    beta, c, f, g, q = sp.beta, sp.c, sp.f, sp.g, sp.q    # Simplify names
    phi_f = lambda p: interp(p, sp.pi_grid, phi)  # Turn phi into a function
    new_phi = np.empty(len(phi))
    for i, pi in enumerate(sp.pi_grid):
        def integrand(x):
            "Integral expression on right-hand side of operator"
            return npmax(x, phi_f(q(x, pi))) * (pi * f(x) + (1 - pi) * g(x))
        integral, error = fixed_quad(integrand, 0, sp.w_max)
        new_phi[i] = (1 - beta) * c + beta * integral
    return new_phi


if __name__ == '__main__':  # If module is run rather than imported

    sp = searchProblem(pi_grid_size=50)
    phi_init = np.ones(len(sp.pi_grid)) 
    w_bar = compute_fixed_point(res_wage_operator, sp, phi_init)

    fig, ax = plt.subplots()
    ax.plot(sp.pi_grid, w_bar, linewidth=2, color='black')
    ax.set_ylim(0, 2)
    ax.grid(axis='x', linewidth=0.25, linestyle='--', color='0.25')
    ax.grid(axis='y', linewidth=0.25, linestyle='--', color='0.25')
    ax.fill_between(sp.pi_grid, 0, w_bar, color='blue', alpha=0.15)
    ax.fill_between(sp.pi_grid, w_bar, 2, color='green', alpha=0.15)
    ax.text(0.42, 1.2, 'reject')
    ax.text(0.7, 1.8, 'accept')
    plt.show()
