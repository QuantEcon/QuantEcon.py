# flake8: noqa
"""
Initialization of the optimize subpackage
"""
from .linprog_simplex import linprog_simplex, solve_tableau, get_solution
from .scalar_maximization import brent_max
from .nelder_mead import nelder_mead
from .root_finding import newton, newton_halley, newton_secant, bisect, brentq
