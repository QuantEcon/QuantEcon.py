"""
Classes for generating and plotting impulse response functions.

@author : David R. Pugh
@date : 2014-10-06

"""
import numpy as np


class ImpulseResponse(object):
    """Base class representing an impulse response function for a Model."""

    def __init__(self, model, N=10, T=100):
        """
        Create an instance of the ImpulseResponse class.

        Parameters
        ----------
        model : model.Model
            Instance of the model.Model class representing a Solow model.
        N : int (default=10)
            Number of points to use for "padding".
        T : int (default=100)
            Length of desired impulse response.

        """
        self.model = model
        self.N = N
        self.T = T

    @property
    def _raw_irf(self):
        """
        Raw model impulse response functions.

        :getter: Return the current raw impulse response functions.
        :type: numpy.ndarray

        """
        # economy is initial in steady state
        k0 = self.model.steady_state

        # generate post-shock trajectory
        soln = self.model.ivp.solve(t0=0.0, y0=k0, h=1.0, T=self.T,
                                    integrator='dop853')

        # compute the irf
        k = soln[:, 1]
        y = self.model.compute_intensive_output(k)[:, np.newaxis]
        c = self.model.compute_consumption(k)[:, np.newaxis]
        i = self.model.compute_actual_investment(k)[:, np.newaxis]

        return np.hstack((soln[:, :2], y, c, i))

    @property
    def _irf_scaling_factor(self):
        """
        Scaling factor used in constructing the impulse response functions.

        :getter: Return the current scaling factor.
        :type: numpy.ndarray

        """
        # extract the relevant parameters
        g = self.model.params['g']
        n = self.model.params['n']
        time = np.linspace(0, self.T, self.T + 1)

        if self.kind == 'per_capita':
            factor = self._padding_scaling_factor[-1] * np.exp(g * time)
        elif self.kind == 'levels':
            factor = self._padding_scaling_factor[-1] * np.exp((n + g) * time)
        else:
            factor = np.ones(self.T + 1)

        return factor.reshape((self.T + 1, 1))

    @property
    def _padding(self):
        """
        Impulse response functions are "padded" for pretty plotting.

        :getter: Return the current "padding" values.
        :type: numpy.ndarray

        """
        return np.hstack((self._padding_time, self._padding_variables))

    @property
    def _padding_scaling_factor(self):
        """
        Scaling factor used in constructing the impulse response function
        "padding".

        :getter: Return the current scaling factor.
        :type: numpy.ndarray

        """
        # extract the relevant parameters
        A0 = self.model.params['A0']
        L0 = self.model.params['L0']
        g = self.model.params['g']
        n = self.model.params['n']

        if self.kind == 'per_capita':
            factor = A0 * np.exp(g * self._padding_time)
        elif self.kind == 'levels':
            factor = A0 * L0 * np.exp((g + n) * self._padding_time)
        else:
            factor = np.ones(self.N)

        return factor.reshape((self.N, 1))

    @property
    def _padding_time(self):
        """
        The independent variable, time, is "padded" using values from -N to 0.

        :getter: Return the current "padding" values.
        :type: numpy.ndarray

        """
        return np.linspace(-self.N, -1, self.N).reshape((self.N, 1))

    @property
    def _padding_variables(self):
        """
        Impulse response functions for endogenous variables are "padded" with
        N periods of steady state values.

        :getter: Return current "padding" values.
        :kind: numpy.ndarray

        """
        # economy is initial in steady state
        k0 = self.model.steady_state
        y0 = self.model.compute_intensive_output(k0)
        c0 = self.model.compute_consumption(k0)
        i0 = self.model.compute_actual_investment(k0)
        intitial_condition = np.array([[k0, y0, c0, i0]])

        # start with N periods of steady state values
        return self._padding_scaling_factor * intitial_condition

    @property
    def impulse(self):
        """
        Dictionary of new parameter values representing an impulse.

        :getter: Return the current impulse dictionary.
        :setter: Set a new impulse dictionary.
        :type: dictionary

        """
        return self._impulse

    @property
    def kind(self):
        """
        The kind of impulse response function to generate. Must be one of:

        * 'levels'
        * 'per_capita'
        * 'efficiency_units'

        :getter: Return the current kind of impulse responses.
        :setter: Set a new value for the kind of impulse responses.
        :type: str

        """
        return self._kind

    @impulse.setter
    def impulse(self, params):
        """Set a new impulse dictionary."""
        self._impulse = self._validate_impulse(params)

    @kind.setter
    def kind(self, value):
        """Set a new value for the kind attribute."""
        self._kind = self._validate_kind(value)

    def _validate_impulse(self, params):
        """Validates the impulse attribute."""
        if not isinstance(params, dict):
            mesg = "ImpulseResponse.impulse must have type dict, not {}."
            raise AttributeError(mesg.format(params.__class__))
        elif not set(params.keys()) < set(self.model.params.keys()):
            mesg = "Invalid parameter included in the impulse dictionary."""
            raise AttributeError(mesg)
        else:
            return params

    @staticmethod
    def _validate_kind(value):
        """Validates the kind attribute."""
        valid_kinds = ['levels', 'per_capita', 'efficiency_units']

        if not isinstance(value, str):
            mesg = "ImpulseResponse.kind must have type str, not {}."
            raise AttributeError(mesg.format(value.__class__))
        elif value not in valid_kinds:
            mesg = "The 'kind' attribute must be in {}."
            raise AttributeError(mesg.format(valid_kinds))
        else:
            return value


def plot_impulse_response(self, variables, param, shock, T, year=2013,
                          color='b', kind='efficiency_units', log=False,
                          reset=True, **fig_kw):
    """
    Plots an impulse response function.

    Parameters
    ----------
    variables : list
        List of variables whose impulse response functions you wish to plot.
        Alternatively, you can plot irfs for all variables by setting variables
        to 'all'.
    param : str
        Model parameter you wish to shock.
    shock : float
        Multiplicative shock to the parameter. Values < 1 correspond to a
        reduction in the current value of the parameter; values > 1 correspond
        to an increase in the current value of the parameter.
    T : float (default=100)
        Length of the impulse response.
    year : int
        Year in which you want the shock to take place. Default is 2013.
    kind : str (default='efficiency_units')
        Whether you want impulse response functions in 'levels', 'per_capita',
        or 'efficiency_units'.
    log : boolean (default=False)
        Whether or not to have logarithmic scales on the vertical axes.
    reset : boolean (default=True)
        Whether or not to reset the original parameters to their pre-shock
        values.

    Returns
    -------
    A list containing:

    fig : object
        An instance of :class:`matplotlib.figure.Figure`.
    axes : list
        A list of instances of :class:`matplotlib.axes.AxesSubplot`.

    """
    # first need to generate and irf
    irf = self.compute_impulse_response(param, shock, T, year, kind, reset)

    # create mapping from variables to column indices
    irf_dict = {'k': irf[:, [0, 1]], 'y': irf[:, [0, 2]], 'c': irf[:, [0, 3]]}

    if variables == 'all':
        variables = irf_dict.keys()

    fig, axes = plt.subplots(len(variables), 1, squeeze=False, **fig_kw)

    for i, var in enumerate(variables):

        # extract the time series
        traj = irf_dict[var]

        # plot the irf
        self.plot_trajectory(traj, color, axes[i, 0])

        # adjust axis limits
        axes[i, 0].set_ylim(0.95 * traj[:, 1].min(), 1.05 * traj[:, 1].max())
        axes[i, 0].set_xlim(year - 10, year + T)

        # y axis labels depend on kind of irfs
        if kind == 'per_capita':
            ti = traj[:, 0] - self.data.index[0].year
            gr = self.params['g']
            axes[i, 0].plot(traj[:, 0], traj[0, 1] * np.exp(gr * ti), 'k--')
            axes[i, 0].set_ylabel(r'$\frac{%s}{L}(t)$' % var.upper(),
                                  rotation='horizontal', fontsize=15,
                                  family='serif')
        elif kind == 'levels':
            ti = traj[:, 0] - self.data.index[0].year
            gr = self.params['n'] + self.params['g']
            axes[i, 0].plot(traj[:, 0], traj[0, 1] * np.exp(gr * ti), 'k--')
            axes[i, 0].set_ylabel('$%s(t)$' % var.upper(),
                                  rotation='horizontal', fontsize=15,
                                  family='serif')
        else:
            axes[i, 0].set_ylabel('$%s(t)$' % var, rotation='horizontal',
                                  fontsize=15, family='serif')

        # adjust location of y-axis label
        axes[i, 0].yaxis.set_label_coords(-0.1, 0.45)

        # log the y-scale for the plots
        if log is True:
            axes[i, 0].set_yscale('log')

    axes[-1, 0].set_xlabel('Year, $t$,', fontsize=15, family='serif')

    return [fig, axes]
