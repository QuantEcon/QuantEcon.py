"""
Filename: linproc.py
Authors: Thomas Sargent, John Stachurski

Provides functions for visualizing scalar ARMA processes.

"""
import numpy as np
from numpy import conj, pi, real
import matplotlib.pyplot as plt
from scipy.signal import dimpulse, freqz, dlsim


class LinearProcess(object):
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
            epsilon_t + theta_1 epsilon_{t-1} + ...  + theta_q epsilon_{t-q}

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
        
            ar_poly(L) X_t = ma_poly(L) epsilon_t 

        where L is the lag operator. To match this, we set
        
            ar_poly = (1, -phi_1, -phi_2,..., -phi_p)
            ma_poly = (1, theta_1, theta_2,..., theta_q) 
            
        In addition, ar_poly must be at least as long as ma_poly.  This can be
        achieved by padding it out with zeros when required.
        """
        # === set up ma_poly === #
        ma_poly = np.asarray(self._theta)
        self.ma_poly = np.insert(ma_poly, 0, 1)      # The array (1, theta)

        # === set up ar_poly === #
        if np.isscalar(self._phi):
            ar_poly = np.array(-self._phi)
        else:
            ar_poly = -np.asarray(self._phi)
        self.ar_poly = np.insert(ar_poly, 0, 1)      # The array (1, -phi)

        # === pad ar_poly with zeros if required === #
        if len(self.ar_poly) < len(self.ma_poly):    
            temp = np.zeros(len(self.ma_poly) - len(self.ar_poly))
            self.ar_poly = np.hstack((self.ar_poly, temp))
        
    def impulse_response(self, impulse_length=30):
        """
        Get the impulse response corresponding to our model.  Returns psi,
        where psi[j] is the response at lag j.  Note: psi[0] is unity.
        """        
        sys = self.ma_poly, self.ar_poly, 1 
        times, psi = dimpulse(sys, n=impulse_length)
        psi = psi[0].flatten()  # Simplify return value into flat array
        return psi

    def spectral_density(self, two_pi=True, res=1200): 
        """
        Compute the spectral density function over [0, pi] if two_pi is False
        and [0, 2 pi] otherwise.  The spectral density is the discrete time
        Fourier transform of the autocovariance function.  In particular,

            f(w) = sum_k gamma(k) exp(-ikw)

        where gamma is the autocovariance function and the sum is over the set
        of all integers.
        """       
        w, h = freqz(self.ma_poly, self.ar_poly, worN=res, whole=two_pi)
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
        sys = self.ma_poly, self.ar_poly, 1
        u = np.random.randn(ts_length, 1)
        vals = dlsim(sys, u)[1]
        return vals.flatten()

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
            plt.show()

    def plot_spectral_density(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.set_title('Spectral density')
        w, spect = self.spectral_density(two_pi=False)  
        ax.semilogy(w, spect)
        ax.set_xlim(0, pi)
        ax.set_ylim(0, np.max(spect))
        ax.set_xlabel('frequency')
        ax.set_ylabel('spectrum')
        if show:
            plt.show()

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
            plt.show()

    def plot_simulation(self, ax=None, show=True):
        if show:
            fig, ax = plt.subplots()
        ax.set_title('Sample path')    
        x_out = self.simulation() 
        ax.plot(x_out)
        ax.set_xlabel('time')
        ax.set_ylabel('state space')
        if show:
            plt.show()

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
        plt.show()


