"""
Solow growth model with Cobb-Douglas aggregate production.

@author : David R. Pugh
@date : 2014-11-27

"""
import numpy as np
import sympy as sym
import pandas as pd

from . import model

# declare key variables for the model
t, X = sym.symbols('t'), sym.DeferredVector('X')
A, k, K, L = sym.symbols('A, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.symbols('g, n, s, alpha, delta')


class CobbDouglasModel(model.Model):

    def __init__(self, params):
        """
        Create an instance of the Solow growth model with Cobb-Douglas
        aggregate production.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.

        """
        cobb_douglas_output = K**alpha * (A * L)**(1 - alpha)
        super(CobbDouglasModel, self).__init__(cobb_douglas_output, params)

    @property
    def steady_state(self):
        r"""
        Steady state value of capital stock (per unit effective labor).

        :getter: Return the current steady state value.
        :type: float

        Notes
        -----
        The steady state value of capital stock (per unit effective labor)
        with Cobb-Douglas production is defined as

        .. math::

            k^* = \bigg(\frac{s}{g + n + \delta}\bigg)^\frac{1}{1-\alpha}

        where `s` is the savings rate, :math:`g + n + \delta` is the effective
        depreciation rate, and :math:`\alpha` is the elasticity of output with
        respect to capital (i.e., capital's share).

        """
        s = self.params['s']
        alpha = self.params['alpha']
        return (s / self.effective_depreciation_rate)**(1 / (1 - alpha))

    def analytic_solution(self, t, k0):
        """
        Compute the analytic solution for the Solow model with Cobb-Douglas
        production technology.

        Parameters
        ----------
        t : ndarray (shape=(T,))
            Array of points at which the solution is desired.
        k0 : (float)
            Initial condition for capital stock (per unit of effective labor)

        Returns
        -------
        analytic_traj : ndarray (shape=t.size, 2)
            Array representing the analytic solution trajectory.

        """
        s = self.params['s']
        alpha = self.params['alpha']

        # lambda governs the speed of convergence
        lmbda = self.effective_depreciation_rate * (1 - alpha)

        # analytic solution for Solow model at time t
        k_t = (((s / (self.effective_depreciation_rate)) * (1 - np.exp(-lmbda * t)) +
                k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

        # combine into a (T, 2) array
        analytic_traj = np.hstack((t[:, np.newaxis], k_t[:, np.newaxis]))

        return analytic_traj


def match_moments(model, data, iso3_code, bounds=None):
    r"""
    Simple calibration scheme for a Solow model with Cobb-Douglas production
    based on data from the Penn World Tables (PWT).

    Parameters
    ----------

    model : solow.CobbDouglasModel
        An instance of the CobbDouglasModel class that you wish to calibrate.
    iso3_code : str
        A valid ISO3 country code. For example, to calibrate the model using
        data for the United States, one would set iso3_code='USA'; to calibrate
        a model using data for Zimbabwe, one would set iso3_code='ZWE'. For a
        complete listing of ISO3 country codes see `wikipedia`_.
    bounds : tuple (default=None)
        Start and end years for the subset of the PWT data to use for
        calibration. Note that start and end years should be specified as
        strings. For example, to calibrate a model using data from 1983 to 2003
        one would set

            bounds=('1983', '2003')

        By default calibration will make use of all available data for the
        specified country.

    .. `wikipedia`: http://en.wikipedia.org/wiki/ISO_3166-1_alpha-3

    """
    # get the PWT data for the iso_code
    tmp_data = data.major_xs(iso3_code)

    # set bounds
    if bounds is None:
        start = tmp_data.index[0]
        end = tmp_data.index[-1]
    else:
        start = bounds[0]
        end = bounds[1]

    # define the data used in the calibration
    output = tmp_data['rgdpna'].loc[start:end]
    capital = tmp_data['rkna'].loc[start:end]
    labor = tmp_data['emp'].loc[start:end]
    labor_share = tmp_data['labsh'].loc[start:end]
    savings_rate = tmp_data['csh_i'].loc[start:end]
    depreciation_rate = tmp_data['delta_k'].loc[start:end]

    # define a time trend variable
    N = tmp_data.index.size
    linear_trend = pd.Series(np.linspace(0, N - 1, N), index=tmp_data.index)
    time_trend = linear_trend.loc[start:end]

    # estimate capital's share of income/output
    capital_share = 1 - labor_share
    alpha = capital_share.mean()

    # compute solow residual (note dependence on alpha!)
    solow_residual = model.evaluate_solow_residual(output, capital, labor)
    technology = solow_residual.loc[start:end]

    # estimate the fraction of output saved
    s = savings_rate.mean()

    # regress log employed persons on linear time trend
    res = pd.ols(y=np.log(labor), x=time_trend)
    n = res.beta[0]
    L0 = np.exp(res.beta[1])

    # regress log TFP on linear time trend
    res = pd.ols(y=np.log(technology), x=time_trend)
    g = res.beta[0]
    A0 = np.exp(res.beta[1])

    # estimate the depreciation rate for total capital
    delta = depreciation_rate.mean()

    # create a dictionary of model parameters
    tmp_params = {'s': s, 'alpha': alpha, 'delta': delta, 'n': n, 'L0': L0,
                  'g': g, 'A0': A0}

    # update the model's parameters
    model.params = tmp_params
