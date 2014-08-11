"""
Author: David R. Pugh

The Lorenz Equations are a system of three coupled, first-order, non-linear
differential equations which describe the trajectory of a particle through
time. The system was originally derived by Lorenz as a model of atmospheric
convection, but the deceptive simplicity of the equations have made them an
often-used example in fields beyond atmospheric physics.

The equations describe the evolution of the spatial variables x, y, and z,
given the governing parameters :math:`\sigma, \beta, \rho`, through the
specification of the time-derivatives of the spatial variables:

.. math::

    \frac{dx}{dt} = \sigma(y − x)
    \frac{dy}{dt} = x(\rho − z) − y
    \frac{dz}{dt} = xy − \beta z

The resulting dynamics are entirely deterministic giving a starting point
:math`(x_0,y_0,z_0)` and a time interval `t`. Though it looks straightforward,
for certain choices of the parameters, the trajectories become chaotic, and the
resulting trajectories display some surprising properties.

"""
import numpy as np
import sympy as sp

# declare endogenous variables
t, x, y, z = sp.var('x, y, z')

# declare model parameters
beta, rho, sigma = sp.var('beta, rho, sigma')

# define symbolic model equations
_x_dot = sigma * (y - x)
_y_dot = x * (rho - z) - y
_z_dot = x * y - beta * z

# define symbolic system and compute the jacobian
X = sp.DeferredVector('X')
change_of_vars = {'x': X[0], 'y': X[1], 'z': X[2]}
_lorenz_system = sp.Matrix([[_x_dot], [_y_dot], [_z_dot]]).subs(change_of_vars)
_lorenz_jacobian = _lorenz_system.jacobian([X[0], X[1], X[2]])

# wrap the symbolic expressions as callable numpy funcs
_args = (t, X, beta, rho, sigma)
_f = sp.lambdify(_args, _lorenz_system,
                 modules=[{'ImmutableMatrix': np.array}, "numpy"])
_jac = sp.lambdify(_args, _lorenz_jacobian,
                   modules=[{'ImmutableMatrix': np.array}, "numpy"])


def f(t, X, beta, rho, sigma):
    """
    Return the Lorenz system.

    Parameters
    ----------
    t : float
        Time
    X : ndarray (float, shape=(3,))
        Endogenous variables of the Lorenz system.
    beta : float
        Model parameter. Should satisfy :math:`0 < \beta`.
    rho : float
        Model parameter. Should satisfy :math:`0 < \rho`.
    sigma : float
        Model parameter. Should satisfy :math:`0 < \sigma`.


    Returns
    -------
    rhs_ode : ndarray (float, shape=(3,))
        Right hand side of the Lorenz system of ODEs.

    """
    rhs_ode = _f(t, X, beta, rho, sigma).ravel()
    return rhs_ode


def jacobian(t, X, beta, rho, sigma):
    """
    Return the Jacobian of the Lorenz system.

    Parameters
    ----------
    t : float
        Time
    X : ndarray (float, shape=(3,))
        Endogenous variables of the Lorenz system.
    beta : float
        Model parameter. Should satisfy :math:`0 < \beta`.
    rho : float
        Model parameter. Should satisfy :math:`0 < \rho`.
    sigma : float
        Model parameter. Should satisfy :math:`0 < \sigma`.

    Returns
    -------
    jac : ndarray (float, shape=(3,3))
        Jacobian of the Lorenz system of ODEs.

    """
    jac = _jac(t, X, beta, rho, sigma)
    return jac
