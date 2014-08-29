"""
Author: David R. Pugh

Lotka-Volterra "Predator-Prey model."

"""
import numpy as np
import sympy as sp

# declare independent and endogenous variables
t, u, v = sp.var('t, u, v')

# declare model parameters
a, b, c, d = sp.var('a, b, c, d')

# define symbolic model equations
_u_dot = a * u - b * u * v
_v_dot = -c * v + d * b * u * v

# define symbolic system and compute the jacobian
X = sp.DeferredVector('X')
change_of_vars = {'u': X[0], 'v': X[1]}

_lotka_volterra_system = sp.Matrix([_u_dot, _v_dot]).subs(change_of_vars)
_lotka_volterra_jacobian = _lotka_volterra_system.jacobian([X[0], X[1]])

# wrap the symbolic expressions as callable numpy funcs
_args = (t, X, a, b, c, d)
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
        Endogenous variables of the Lotka-Volterra system. Ordering is
        `X = [u, v]` where `u` is the number of prey and `v` is the number of
        predators.
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
    rhs_ode : ndarray (float, shape=(2,))
        Right hand side of the Lotka-Volterra system of ODEs.

    """
    rhs_ode = _f(t, X, a, b, c, d).ravel()
    return rhs_ode


def jacobian(t, X, a, b, c, d):
    """
    Return the Lotka-Voltera Jacobian matrix.

    Parameters
    ----------
    t : float
        Time
    X : ndarray (float, shape=(2,))
        Endogenous variables of the Lotka-Volterra system. Ordering is
        `X = [u, v]` where `u` is the number of prey and `v` is the number of
        predators.
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
    jac = _jac(t, X, a, b, c, d)
    return jac
