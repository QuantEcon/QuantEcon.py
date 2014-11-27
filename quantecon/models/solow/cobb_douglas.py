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
t, X = sym.var('t'), sym.DeferredVector('X')
A, k, K, L = sym.var('A, k, K, L')

# declare required model parameters
g, n, s, alpha, delta = sym.var('g, n, s, alpha, delta')


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


def analytic_solution(cls, t, k0):
    """
    Compute the analytic solution for the Solow model with Cobb-Douglas
    production technology.

    Parameters
    ----------
    cls : object
        Instance of the `solow.CobbDouglasModel` class.
    t : ndarray (shape=(T,))
        Array of points at which the solution is desired.
    k0 : (float)
        Initial condition for capital stock (per unit of effective labor)

    Returns
    -------
    analytic_traj : ndarray (shape=t.size, 2)
        Array representing the analytic solution trajectory.

    """
    s = cls.params['s']
    alpha = cls.params['alpha']

    # lambda governs the speed of convergence
    lmbda = cls.effective_depreciation_rate * (1 - alpha)

    # analytic solution for Solow model at time t
    k_t = (((s / (cls.effective_depreciation_rate)) * (1 - np.exp(-lmbda * t)) +
            k0**(1 - alpha) * np.exp(-lmbda * t))**(1 / (1 - alpha)))

    # combine into a (T, 2) array
    analytic_traj = np.hstack((t[:, np.newaxis], k_t[:, np.newaxis]))

    return analytic_traj


def analytic_steady_state(cls):
    """
    Steady-state level of capital stock (per unit effective labor).

    Parameters
    ----------
    cls : object
        Instance of the `solow.CobbDouglasModel` class.

    Returns
    -------
    kstar : float
        Steady state value of capital stock (per unit effective labor).

    """
    s = cls.params['s']
    alpha = cls.params['alpha']

    k_star = (s / cls.effective_depreciation_rate)**(1 / (1 - alpha))

    return k_star


def calibrate(cls, data, iso3_code, bounds=None):
    r"""
    Calibrates a Solow model with Cobb-Douglas production using data from the
    Penn World Tables (PWT).

    Parameters
    ----------

    model : solow.Model
        An instance of the SolowModel class that you wish to calibrate.
    iso3_code : str
        A valid ISO3 country code. For example, to calibrate the model using
        data for the United States, one would set iso3_code='USA'; to calibrate
        a model using data for Zimbabwe, one would set iso3_code='ZWE'. For a
        complete listing of ISO3 country codes see `wikipedia`_.
    bounds:    (tuple) Start and end years for the subset of the PWT data
               to use for calibration.

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

    # estimate capital's share of income/output
    alpha = (1 - tmp_data.labsh.loc[start:end]).mean()

    # estimate the fraction of output saved
    s = tmp_data.csh_i.loc[start:end].mean()

    # regress log employed persons on linear time trend
    N = tmp_data.index.size
    trend = pd.Series(np.linspace(0, N - 1, N), index=tmp_data.index)
    res = pd.ols(y=np.log(tmp_data.emp.loc[start:end]),
                 x=trend.loc[start:end])
    n = res.beta[0]
    L0 = np.exp(res.beta[1])

    # estimate the technology growth rate

    # adjust measure of TFP
    model.data['atfpna'] = (tmp_data.rtfpna**(1 / tmp_data.labsh) *
                            tmp_data.hc)

    # regress log TFP on linear time trend
    res = pd.ols(y=np.log(tmp_data.atfpna.loc[start:end]),
                 x=trend.loc[start:end])
    g = res.beta[0]
    A0 = np.exp(res.beta[1])

    # estimate the depreciation rate for total capital
    delta = tmp_data.delta_k.loc[start:end].mean()

    # create a dictionary of model parameters
    tmp_params = {'s': s, 'alpha': alpha, 'delta': delta, 'n': n, 'L0': L0,
                  'g': g, 'A0': A0}

    # update the model's parameters
    model.params.update(tmp_params)
