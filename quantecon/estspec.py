"""
Filename: estspec.py
Authors: Thomas Sargent, John Stachurski 

Functions for working with periodograms of scalar data.
"""

from __future__ import division, print_function  # Omit for Python 3.x
import numpy as np
from numpy.fft import fft
from pandas import ols, Series

def smooth(x, window_len=7, window='hanning'):
    """
    Smooth the data in x using convolution with a window of requested size
    and type.

    Parameters:

        * x is a flat NumPy array --- the data to smooth

        * window_len is an odd integer --- the length of the window

        * window is a string giving the window type 
          ('flat', 'hanning', 'hamming', 'bartlett' or 'blackman')

    Application of the smoothing window at the top and bottom of x is done by
    reflecting x around these points to extend it sufficiently in each
    direction.

    """
    if len(x) < window_len:
        raise ValueError, "Input vector length must be at least window length."

    if window_len < 3:
        raise ValueError, "Window length must be at least 3."

    if not window_len % 2:  # window_len is even
        window_len +=1
        print("Window length reset to {}".format(window_len))

    windows = {'hanning': np.hanning,
               'hamming': np.hamming, 
               'bartlett': np.bartlett,
               'blackman': np.blackman}

    # === reflect x around x[0] and x[-1] prior to convolution === #
    k = int(window_len / 2)
    xb = x[:k]   # First k elements
    xt = x[-k:]  # Last k elements
    s = np.concatenate((xb[::-1], x, xt[::-1]))
    
    # === select window values === #
    if window == 'flat':  
        w = np.ones(window_len)  # moving average
    else:
        try:
            w = windows[window](window_len)
        except KeyError:
            print("Unrecognized window type.  Defaulting to 'hanning'.")
            w = windows['hanning'](window_len)

    return np.convolve(w / w.sum(), s, mode='valid')


def periodogram(x, window=None, window_len=7):
    """
    Computes the periodogram 

        I(w) = (1 / n) | sum_{t=0}^{n-1} x_t e^{itw} |^2

    at the Fourier frequences w_j := 2 pi j / n, j = 0, ..., n - 1, using the
    fast Fourier transform.  Only the frequences w_j in [0, pi] and
    corresponding values I(w_j) are returned.  If a window type is given then
    smoothing is performed.

        * x is a flat NumPy array --- the time series data

        * window is a string giving the window type 
          ('flat', 'hanning', 'hamming', 'bartlett' or 'blackman')

        * window_len is an odd integer --- the length of the window

    """
    n = len(x)
    I_w = np.abs(fft(x))**2 / n
    w = 2 * np.pi * np.arange(n) / n
    w, I_w = w[:int(n/2)+1], I_w[:int(n/2)+1]  # Take only values on [0, pi]
    if window:
        I_w = smooth(I_w, window_len=window_len, window=window)
    return w, I_w


def ar_periodogram(x, window='hanning', window_len=7):
    """
    Compute periodogram from data x, using prewhitening, smoothing and
    recoloring.  The data is fitted to an AR(1) model for prewhitening,
    and the residuals are used to compute a first-pass periodogram with
    smoothing.  The fitted coefficients are then used for recoloring.

    Parameters:

        * x is a NumPy array containing time series data
        * window is a string indicating window type 
        * window_len is an odd integer

    See the periodogram function documentation for more details on the window
    arguments.
    """              
    # === run regression === #
    x_current, x_lagged = x[1:], x[:-1]                       # x_t and x_{t-1}
    x_current, x_lagged = Series(x_current), Series(x_lagged) # pandas series
    results = ols(y=x_current, x=x_lagged, intercept=True, nw_lags=1)
    e_hat = results.resid.values
    phi = results.beta['x']

    # === compute periodogram on residuals === #
    w, I_w = periodogram(e_hat, window=window, window_len=window_len) 

    # === recolor and return === #
    I_w = I_w  / np.abs(1 - phi * np.exp(1j * w))**2  
    return w, I_w
