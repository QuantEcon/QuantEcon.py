"""
Author: David R. Pugh

Lotka-Volterra "Predator-Prey model."

"""
import numpy as np
import sympy as sp

# declare endogenous variables
X = sp.DeferredVector('X')

# declare model parameters
a, b, c, d = sp.var('a, b, c, d')

# define symbolic model equations
_u_dot = a * X[0] - b * X[0] * X[1]
_v_dot = -c * X[1] + d * b * X[0] * X[1]

# define symbolic system and compute the jacobian
_lotka_volterra_system = sp.Matrix([_u_dot, _v_dot])
_lotka_volterra_jacobian = _lotka_volterra_system.jacobian([X[0], X[1]])

# wrap the symbolic expressions as callable numpy funcs
_args = (X, a, b, c, d)
_f = sp.lambdify(_args, _lotka_volterra_system,
                 modules=[{'ImmutableMatrix': np.array}, "numpy"])
_jac = sp.lambdify(_args, _lotka_volterra_jacobian,
                   modules=[{'ImmutableMatrix': np.array}, "numpy"])


def f(t, X, a, b, c, d):
    """
    Return the Lotka-Voltera system

    Parameters
    ----------
    t : float
        Time
    X : ndarray (float, shape=(2,))
        Endogenous variables of the Lotka-Volterra system.
    a : float
        Natural growth rate of prey in the absence of predators.
    b : float
        Natural death rate of prey due to predation.
    c : float
        Natural death rate of predators, due to absence of prey.
    d : float
        Factor describing how many caught prey is necessary to create a new
        predator.

    Returns
    -------
    f : ndarray (float, shape=(2,))
        RHS of the Lotka-Volterra system of ODEs.

    """
    f = _f(X, a, b, c, d).ravel()
    return f


def jacobian(t, X, a, b, c, d):
    """
    Return the Lotka-Voltera Jacobian matrix.

    Parameters
    ----------
    t : float
        Time
    X : ndarray (float, shape=(2,))
        Endogenous variables of the Lotka-Volterra system.
    a : float
        Natural growth rate of prey in the absence of predators.
    b : float
        Natural death rate of prey due to predation.
    c : float
        Natural death rate of predators, due to absence of prey.
    d : float
        Factor describing how many caught prey is necessary to create a new
        predator.

    Returns
    -------
    jac : ndarray (float, shape=(2,2))
        Jacobian of the Lotka-Volterra system of ODEs.

    """
    jac = _jac(X, a, b, c, d)
    return jac
