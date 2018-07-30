"""
Initialization of the optimize subpackage
"""

from .scalar_maximization import brent_max
from .root_finding import newton, newton_halley, newton_secant, bisect, brentq
