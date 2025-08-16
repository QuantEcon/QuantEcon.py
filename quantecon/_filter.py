"""

function for filtering

"""
import numpy as np


def hamilton_filter(data, h, p=None):
    r"""
    This function applies "Hamilton filter" to the data

    http://econweb.ucsd.edu/~jhamilto/hp.pdf

    Parameters
    ----------
    data : array or dataframe
    h : integer
        Time horizon that we are likely to predict incorrectly.
        Original paper recommends 2 for annual data, 8 for quarterly data,
        24 for monthly data.
    p : integer (optional)
        If supplied, it is p in the paper. Number of lags in regression.
        If not supplied, random walk process is assumed.

    Returns
    -------
    cycle : array of cyclical component
    trend : trend component

    Notes
    -----
    For seasonal data, it's desirable for p and h to be integer multiples of
    the number of obsevations in a year. E.g. for quarterly data, h = 8 and p =
    4 are recommended.

    """
    # transform data to array
    y = np.asarray(data, float)
    # sample size
    T = len(y)

    if p is not None:  # if p is supplied
        # construct X matrix of lags
        X = np.ones((T-p-h+1, p+1))
        for j in range(1, p+1):
            X[:, j] = y[p-j:T-h-j+1:1]

        # do OLS regression
        b = np.linalg.solve(X.transpose()@X, X.transpose()@y[p+h-1:T])
        # trend component (`nan` for the first p+h-1 period)
        trend = np.append(np.zeros(p+h-1)+np.nan, X@b)
        # cyclical component
        cycle = y - trend
    else:  # if p is not supplied (random walk)
        cycle = np.append(np.zeros(h)+np.nan, y[h:T] - y[0:T-h])
        trend = y - cycle
    return cycle, trend
