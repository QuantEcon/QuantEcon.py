"""
Filename: linear_process.py
Authors: Thomas Sargent, Doc-Jin Jang, Jeoung-hun Choi, John Stachurski
Date: May 2013
"""
import numpy as np
from numpy import conj, pi, real
import matplotlib.pyplot as plt
from scipy.signal import dimpulse, freqz, dlsim


class linearProcess(object):
    """
    This class provides functions for working with scalar ARMA processes.  In
    particular, it defines methods for computing and plotting the
    autocovariance function, the spectral density, the impulse-response
    function and simulated time series.

    """
    
    def __init__(self, phi, theta=0, sigma=1) :
        """
        This class represents scalar ARMA(p, q) processes.  The parameters phi
        and theta can be NumPy arrays, array-like sequences (lists, tuples) or
        scalars.

        If phi and theta are scalars, then the model is
        understood to be 
        
            X_t = phi X_{t-1} + epsilon_t + theta epsilon_{t-1}  
            
        where {epsilon_t} is a white noise process with standard deviation
        sigma.  If phi and theta are arrays or sequences, then the
        interpretation is the ARMA(p, q) model 

            X_t = phi_1 X_{t-1} + ... + phi_p X_{t-p} + 
                epsilon_t + theta_1 epsilon_{t-1} + ... + theta_q epsilon_{t-q}

        where

            * phi = (phi_1, phi_2,..., phi_p)
            * theta = (theta_1, theta_2,..., theta_q)
            * sigma is a scalar, the standard deviation of the white noise

        """
        self._phi, self._theta = phi, theta
        self.sigma = sigma
        self.set_params()  

    def get_phi(self):
        return self._phi

    def get_theta(self):
        return self._theta

    def set_phi(self, new_value):
        self._phi = new_value
        self.set_params()

    def set_theta(self, new_value):
        self._theta = new_value
        self.set_params()

    phi = property(get_phi, set_phi)
    theta = property(get_theta, set_theta)

    def set_params(self):
        """
        Internally, scipy.signal works with systems of the form 
        
            den(L) X_t = num(L) epsilon_t 

        where L is the lag operator. To match this, we set
        
            den = (1, -phi_1, -phi_2,..., -phi_p)
            num = (1, theta_1, theta_2,..., theta_q) 
            
        In addition, den must be at least as long as num.  This can be
        achieved by padding it out with zeros when required.
        """
        num = np.asarray(self._theta)
        if np.isscalar(self._phi):
            den = np.array(-self._phi)
        else:
            den = -np.asarray(self._phi)
        self.num = np.insert(num, 0, 1)      # The array (1, theta)
        self.den = np.insert(den, 0, 1)      # The array (1, -phi)
        if len(self.den) < len(self.num):    # Pad den with zeros if necessary
            temp = np.zeros(len(self.num) - len(self.den))
            self.den = np.hstack((self.den, temp))
        
    def impulse_response(self, impulse_length=30):
        """
        Get the impulse response corresponding to our model.  Returns psi,
        where psi[j] is the response at lag j.  Note: psi[0] is unity.
        """        
        sys = self.num, self.den, 1 
        times, psi = dimpulse(sys, n=impulse_length)
        psi = psi[0].flatten()  # Simplify return value into flat array
        return psi

    def spectral_density(self, domain_max=2*pi, resolution=1e5) :
        """
        Compute the spectral density function over domain [0, domain_max].
        The spectral density is the discrete time Fourier transform of the
        autocovariance function.  In particular,

            f(w) = sum_k gamma(k) exp(-ikw)

        where gamma is the autocovariance function and the sum is over k in Z,
        the set of all integers.
        """       
        w = np.linspace(0, domain_max, resolution)
        h = freqz(self.num, self.den, w)[1]
        spect = h * conj(h) * self.sigma**2 
        return w, spect

    def autocovariance(self, num_autocov=16) :
        """
        Compute the autocovariance function over the integers
        range(num_autocov) using the spectral density and the inverse Fourier
        transform.
        """
        spect = self.spectral_density()[1]
        acov = np.fft.ifft(spect).real
        return acov[:num_autocov]  # num_autocov should be <= len(acov) / 2

    def simulation(self, ts_length=90) :
        " Compute a simulated sample path. "        
        sys = self.num, self.den, 1
        u = np.random.randn(ts_length, 1)
        return dlsim(sys, u)[1]

    def plot_impulse_response(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.set_title('Impulse response')
        yi = self.impulse_response()
        ax.stem(range(len(yi)), yi)
        ax.set_xlim(xmin=(-0.5))
        ax.set_ylim(min(yi)-0.1,max(yi)+0.1)
        ax.set_xlabel('time')
        ax.set_ylabel('response')
        if show:
            fig.show()

    def plot_spectral_density(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.set_title('Spectral density')
        w, spect = self.spectral_density(domain_max=pi)  
        ax.semilogy(w, spect)
        ax.set_xlim(0, pi)
        ax.set_ylim(0, np.max(spect))
        ax.set_xlabel('frequency')
        ax.set_ylabel('spectrum')
        if show:
            fig.show()

    def plot_autocovariance(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.set_title('Autocovariance')
        acov = self.autocovariance() 
        ax.stem(range(len(acov)), acov)
        ax.set_xlim(-0.5, len(acov) - 0.5)
        ax.set_xlabel('time')
        ax.set_ylabel('autocovariance')     
        if show:
            fig.show()

    def plot_simulation(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.set_title('Sample path')    
        x_out = self.simulation() 
        ax.plot(x_out)
        ax.set_xlabel('time')
        ax.set_ylabel('state space')
        if show:
            fig.show()

    def quad_plot(self) :
        """
        Plots the impulse response, spectral_density, autocovariance, and one
        realization of the process.
        """
        num_rows, num_cols = 2, 2
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4)
        self.plot_impulse_response(axes[0, 0], show=False)
        self.plot_spectral_density(axes[0, 1], show=False)
        self.plot_autocovariance(axes[1, 0], show=False)
        self.plot_simulation(axes[1, 1], show=False)
        fig.show()


