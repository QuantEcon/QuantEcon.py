"""
Initialization of the optimize subpackage
"""

from .scalar_maximization import brent_max
from .multivar_maximization import maximize, nelder_mead_algorithm
from .root_finding import newton, newton_halley, newton_secant, bisect, brentq
