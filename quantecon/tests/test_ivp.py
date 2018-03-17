"""
Tests for ivp.py

"""
import nose

import numpy as np

from .. import ivp


# use the Solow Model with Cobb-Douglas production as test case
def solow_model(t, k, g, n, s, alpha, delta):
    """
    Equation of motion for capital stock (per unit effective labor).

    Parameters
    ----------
    t : float
        Time
    k : ndarray (float, shape=(1,))
        Capital stock (per unit of effective labor)
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Elasticity of output with respect to capital stock. Must satisfy
        :math:`0 < alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy :math:`0 < \delta`.

    Returns
    -------
    k_dot : ndarray (float, shape(1,))
        Time derivative of capital stock (per unit effective labor).

    """
    k_dot = s * k**alpha - (g + n + delta) * k
    return k_dot


def solow_jacobian(t, k, g, n, s, alpha, delta):
    """
    Jacobian matrix for the Solow model.

    Parameters
    ----------
    t : float
        Time
    k : ndarray (float, shape=(1,))
        Capital stock (per unit of effective labor)
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Elasticity of output with respect to capital stock. Must satisfy
        :math:`0 < alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy :math:`0 < \delta`.

    Returns
    -------
    jac : ndarray (float, shape(1,))
        Time derivative of capital stock (per unit effective labor).

    """
    jac = s * alpha * k**(alpha - 1) - (g + n + delta)
    return jac


def solow_steady_state(g, n, s, alpha, delta):
    """
    Steady-state level of capital stock (per unit effective labor).

    Parameters
    ----------
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Elasticity of output with respect to capital stock. Must satisfy
        :math:`0 < alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy :math:`0 < \delta`.

    Returns
    -------
    kstar : float
        Steady state value of capital stock (per unit effective labor).

    """
    k_star = (s / (n + g + delta))**(1 / (1 - alpha))
    return k_star


def solow_analytic_solution(t, k0, g, n, s, alpha, delta):
    """
    Analytic solution for the path of capital stock (per unit effective labor).

    Parameters
    ----------
    t : ndarray(float, shape=(1,))
        Time
    k : ndarray (float, shape=(1,))
        Capital stock (per unit of effective labor)
    g : float
        Growth rate of technology.
    n : float
        Growth rate of the labor force.
    s : float
        Savings rate. Must satisfy `0 < s < 1`.
    alpha : float
        Elasticity of output with respect to capital stock. Must satisfy
        :math:`0 < alpha < 1`.
    delta : float
        Depreciation rate of physical capital. Must satisfy :math:`0 < \delta`.

    Returns
    -------
    soln : ndarray (float, shape(t.size, 2))
        Trajectory describing the analytic solution of the model.

    """
    # lambda governs the speed of convergence
    lmbda = (n + g + delta) * (1 - alpha)

    # analytic solution for Solow model at time t
    k_t = (((s / (n + g + delta)) * (1 - np.exp(-lmbda * t)) +
            k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

    # combine into a (t.size, 2) array
    analytic_traj = np.hstack((t[:, np.newaxis], k_t[:, np.newaxis]))

    return analytic_traj

# create an instance of the IVP class
valid_params = (0.02, 0.02, 0.15, 0.33, 0.05)
model = ivp.IVP(f=solow_model,
                jac=solow_jacobian)

model.f_params = valid_params
model.jac_params = valid_params


# helper functions for conducting tests
def _compute_fixed_length_solns(model, t0, k0):
    """Returns a dictionary of fixed length solution trajectories."""

    # storage containter for integration results
    results = {}

    for integrator in ['dopri5', 'dop853', 'vode', 'lsoda']:

        # tighten tolerances so tests don't fail due to numerical issues
        discrete_soln = model.solve(t0, k0, h=1e0, T=1e3,
                                    integrator=integrator,
                                    atol=1e-14, rtol=1e-11)

        # store the result
        results[integrator] = discrete_soln

    return results


def _termination_condition(t, k, g, n, s, alpha, delta):
        """Terminate solver when we get close to steady state."""
        diff = k - solow_steady_state(g, n, s, alpha, delta)
        return diff


def _compute_variable_length_solns(model, t0, k0, g, tol):
    """Returns a dictionary of variable length solution trajectories."""

    # storage containter for integration results
    results = {}

    for integrator in ['dopri5', 'dop853', 'vode', 'lsoda']:

        # tighten tolerances so tests don't fail due to numerical issues
        discrete_soln = model.solve(t0, k0, h=1e0, g=g, tol=tol,
                                    integrator=integrator,
                                    atol=1e-14, rtol=1e-11)

        # store the result
        results[integrator] = discrete_soln

    return results


# testing functions
def test_solve_args():
    """Testing arguments passed to the IVP.solve method."""
    # g and tol must be passed together!
    with nose.tools.assert_raises(ValueError):
        t0, k0 = 0, np.array([5.0])
        model.solve(t0, k0, g=_termination_condition)


def test_solve_fixed_trajectory():
    """Testing computation of fixed length solution trajectory."""

    # compute some fixed length trajectories
    t0, k0 = 0, np.array([5.0])
    results = _compute_fixed_length_solns(model, t0, k0)

    # conduct the tests
    for integrator, numeric_solution in results.items():
        ti = numeric_solution[:, 0]
        analytic_solution = solow_analytic_solution(ti, k0, *valid_params)
        np.testing.assert_allclose(numeric_solution, analytic_solution)


def test_solve_variable_trajectory():
    """Testing computation of variable length solution trajectory."""

    # compute some variable length trajectories
    t0, k0, tol = 0, np.array([5.0]), 1e-3
    results = _compute_variable_length_solns(model, t0, k0,
                                             g=_termination_condition, tol=tol)

    # conduct the tests
    for integrator, numeric_solution in results.items():
        ti = numeric_solution[:, 0]
        analytic_solution = solow_analytic_solution(ti, k0, *valid_params)

        # test accuracy of solution
        np.testing.assert_allclose(numeric_solution, analytic_solution)

        # test termination condition
        diff = numeric_solution[-1, 1] - solow_steady_state(*valid_params)
        nose.tools.assert_less_equal(diff, tol)


def test_interpolation():
    """Testing parameteric B-spline interpolation methods."""

    # compute some fixed length trajectories
    t0, k0 = 0, np.array([5.0])
    results = _compute_fixed_length_solns(model, t0, k0)

    # conduct the tests
    for integrator, numeric_solution in results.items():

        # define an array of interpolation points
        N, T = 1000, numeric_solution[:, 0][-1]
        ti = np.linspace(t0, T, N)

        # used highest order B-spline interpolation available
        interp_solution = model.interpolate(numeric_solution, ti, k=3, ext=2)

        analytic_solution = solow_analytic_solution(ti, k0, *valid_params)
        np.testing.assert_allclose(interp_solution, analytic_solution)


def test_compute_residual():
    """Testing computation of solution residual."""

    # compute some fixed length trajectories
    t0, k0 = 0, np.array([5.0])
    results = _compute_fixed_length_solns(model, t0, k0)

    # conduct the tests
    for integrator, numeric_solution in results.items():

        # define an array of interpolation points
        N, T = 1000, numeric_solution[:, 0][-1]
        tmp_grid_pts = np.linspace(t0, T, N)

        # used highest order B-spline interpolation available
        tmp_residual = model.compute_residual(numeric_solution,
                                              tmp_grid_pts,
                                              k=5)

        expected_residual = np.zeros((N, 2))
        actual_residual = tmp_residual
        np.testing.assert_almost_equal(expected_residual, actual_residual)
