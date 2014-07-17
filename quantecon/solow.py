import ivp


class Model(ivp.IVP):
    """Base class representing a continuous time Solow growth model."""

    def __init__(self, k_dot, jacobian, params):
        """
        Creates an instance of the Solow model.

        Arguments
        ----------
        k_dot : callable, ``k_dot(t, k, params)``
            Equation of motion for capital (per person/effective person). The
            independent variable, `t`, is time; `k`, is capital (per person/
            effective person); `params` is a dictionary of model parameters.
        jacobian : callable, ``jacobian(t, k, params)``
            The derivative of the equation of motion for capital (per person/
            effective person) with respect to `k`. The independent variable, t,
            is time; k, (per person/effective person); `params` is a dictionary
            of model parameters.
        params : dict
            Dictionary of model parameters. Standard parameters for a Solow
            growth model are:

            - `g`: Growth rate of technology (rate of technological progress).
            - `n`: Growth rate of labor force.
            - `s`: Savings rate. Must satisfy ``0 < s < 1``.
            - :math:`\delta`: Depreciation rate of physical capital. Must
            satisfy :math:`0 < \delta`.

            Only other model parameters will be the parameters of some
            production function.

        """
        super(Model, self).__init__(k_dot, jacobian, (params,))


def cobb_douglas_analytic_solution(k0, t, g, n, s, alpha, delta):
    """
    Analytic solution for the Solow model with Cobb-Douglas production.

    Arguments
    ---------
        k0 : float
            Initial value for capital (per person/effective person)
        t : array_like (float, shape=(T,))
            Array of points at which the solution is desired.
        g : float
            Growth rate of technology.
        n : float
            Growth rate of the labor force.
        s : float
            Savings rate. Must satisfy ``0 < s < 1``.
        alpha : float
            Elasticity of output with respect to capital. Must satisfy
            :math:`0 < \alpha < 1`.
        delta : float
            Depreciation rate of physical capital. Must satisfy
            :math:`0 < \delta`.

    Returns
    -------
        analytic_traj : array_like (float, shape=(T,2))
            Array representing the analytic solution trajectory.

    """
    # speed of convergence
    lmbda = (n + g + delta) * (1 - alpha)

    # analytic solution for Solow model at time t
    k_t   = (((s / (n + g + delta)) * (1 - np.exp(-lmbda * t)) +
              k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

    # combine into a (T, 2) array
    analytic_traj = np.hstack((t[:,np.newaxis], k_t[:,np.newaxis]))

    return analytic_traj
