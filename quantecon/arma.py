"""
Provides functions for working with and visualizing scalar ARMA processes.

TODO: 1. Fix warnings concerning casting complex variables back to floats

"""
import numpy as np
from .util import check_random_state


class ARMA:
    r"""
    This class represents scalar ARMA(p, q) processes.

    If phi and theta are scalars, then the model is
    understood to be

    .. math::

        X_t = \phi X_{t-1} + \epsilon_t + \theta \epsilon_{t-1}

    where :math:`\epsilon_t` is a white noise process with standard
    deviation :math:`\sigma`.  If phi and theta are arrays or sequences,
    then the interpretation is the ARMA(p, q) model

    .. math::

        X_t = \phi_1 X_{t-1} + ... + \phi_p X_{t-p} +

        \epsilon_t + \theta_1 \epsilon_{t-1} + ...  +
        \theta_q \epsilon_{t-q}

    where

        * :math:`\phi = (\phi_1, \phi_2,..., \phi_p)`
        * :math:`\theta = (\theta_1, \theta_2,..., \theta_q)`
        * :math:`\sigma` is a scalar, the standard deviation of the
          white noise

    Parameters
    ----------
    phi : scalar or iterable or array_like(float)
        Autocorrelation values for the autocorrelated variable.
        See above for explanation.
    theta : scalar or iterable or array_like(float)
        Autocorrelation values for the white noise of the model.
        See above for explanation
    sigma : scalar(float)
        The standard deviation of the white noise

    Attributes
    ----------
    phi, theta, sigma : see Parmeters
    ar_poly : array_like(float)
        The polynomial form that is needed by scipy.signal to do the
        processing we desire.  Corresponds with the phi values
    ma_poly : array_like(float)
        The polynomial form that is needed by scipy.signal to do the
        processing we desire.  Corresponds with the theta values

    """
    def __init__(self, phi, theta=0, sigma=1):
        self._phi, self._theta = phi, theta
        self.sigma = sigma
        self.set_params()

    def __repr__(self):
        m = "ARMA(phi=%s, theta=%s, sigma=%s)"
        return m % (self.phi, self.theta, self.sigma)

    def __str__(self):
        m = "An ARMA({p}, {q}) process"
        p = np.asarray(self.phi).size
        q = np.asarray(self.theta).size
        return m.format(p=p, q=q)

    # Special latex print method for working in notebook
    def _repr_latex_(self):
        m = r"$X_t = "
        phi = np.atleast_1d(self.phi)
        theta = np.atleast_1d(self.theta)
        rhs = ""
        for (tm, phi_p) in enumerate(phi):
            # don't include terms if they are equal to zero
            if abs(phi_p) > 1e-12:
                rhs += r"%+g X_{t-%i}" % (phi_p, tm+1)

        if rhs[0] == "+":
            rhs = rhs[1:]  # remove initial `+` if phi_1 was positive

        rhs += r" + \epsilon_t"

        for (tm, th_q) in enumerate(theta):
            # don't include terms if they are equal to zero
            if abs(th_q) > 1e-12:
                rhs += r"%+g \epsilon_{t-%i}" % (th_q, tm+1)

        return m + rhs + "$"

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, new_value):
        self._phi = new_value
        self.set_params()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, new_value):
        self._theta = new_value
        self.set_params()

    def set_params(self):
        r"""
        Internally, scipy.signal works with systems of the form

        .. math::

            ar_{poly}(L) X_t = ma_{poly}(L) \epsilon_t

        where L is the lag operator. To match this, we set

        .. math::

            ar_{poly} = (1, -\phi_1, -\phi_2,..., -\phi_p)

            ma_{poly} = (1, \theta_1, \theta_2,..., \theta_q)

        In addition, ar_poly must be at least as long as ma_poly.
        This can be achieved by padding it out with zeros when required.

        """
        # === set up ma_poly === #
        ma_poly = np.asarray(self._theta)
        self.ma_poly = np.insert(ma_poly, 0, 1)  # The array (1, theta)

        # === set up ar_poly === #
        if np.isscalar(self._phi):
            ar_poly = np.array(-self._phi)
        else:
            ar_poly = -np.asarray(self._phi)
        self.ar_poly = np.insert(ar_poly, 0, 1)  # The array (1, -phi)

        # === pad ar_poly with zeros if required === #
        if len(self.ar_poly) < len(self.ma_poly):
            temp = np.zeros(len(self.ma_poly) - len(self.ar_poly))
            self.ar_poly = np.hstack((self.ar_poly, temp))

    def impulse_response(self, impulse_length=30):
        """
        Get the impulse response corresponding to our model.

        Returns
        -------
        psi : array_like(float)
            psi[j] is the response at lag j of the impulse response.
            We take psi[0] as unity.

        """
        from scipy.signal import dimpulse
        sys = self.ma_poly, self.ar_poly, 1
        times, psi = dimpulse(sys, n=impulse_length)
        psi = psi[0].flatten()  # Simplify return value into flat array

        return psi

    def spectral_density(self, two_pi=True, res=1200):
        r"""
        Compute the spectral density function.  The spectral density is
        the discrete time Fourier transform of the autocovariance
        function.  In particular,

        .. math::

            f(w) = \sum_k \gamma(k) \exp(-ikw)

        where gamma is the autocovariance function and the sum is over
        the set of all integers.

        Parameters
        ----------
        two_pi : Boolean, optional
            Compute the spectral density function over :math:`[0, \pi]` if
            two_pi is False and :math:`[0, 2 \pi]` otherwise.  Default value is
            True
        res : scalar or array_like(int), optional(default=1200)
            If res is a scalar then the spectral density is computed at
            `res` frequencies evenly spaced around the unit circle, but
            if res is an array then the function computes the response
            at the frequencies given by the array

        Returns
        -------
        w : array_like(float)
            The normalized frequencies at which h was computed, in
            radians/sample
        spect : array_like(float)
            The frequency response

        """
        from scipy.signal import freqz
        w, h = freqz(self.ma_poly, self.ar_poly, worN=res, whole=two_pi)
        spect = h * np.conj(h) * self.sigma**2

        return w, spect

    def autocovariance(self, num_autocov=16):
        """
        Compute the autocovariance function from the ARMA parameters
        over the integers range(num_autocov) using the spectral density
        and the inverse Fourier transform.

        Parameters
        ----------
        num_autocov : scalar(int), optional(default=16)
            The number of autocovariances to calculate

        """
        spect = self.spectral_density()[1]
        acov = np.fft.ifft(spect).real

        # num_autocov should be <= len(acov) / 2
        return acov[:num_autocov]

    def simulation(self, ts_length=90, random_state=None):
        """
        Compute a simulated sample path assuming Gaussian shocks.

        Parameters
        ----------
        ts_length : scalar(int), optional(default=90)
            Number of periods to simulate for

        random_state : int or np.random.RandomState, optional
            Random seed (integer) or np.random.RandomState instance to set
            the initial state of the random number generator for
            reproducibility. If None, a randomly initialized RandomState is
            used.

        Returns
        -------
        vals : array_like(float)
            A simulation of the model that corresponds to this class

        """
        from scipy.signal import dlsim
        random_state = check_random_state(random_state)

        sys = self.ma_poly, self.ar_poly, 1
        u = random_state.randn(ts_length, 1) * self.sigma
        vals = dlsim(sys, u)[1]

        return vals.flatten()
