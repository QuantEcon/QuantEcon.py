from numba import njit

@njit
def interp(x, xp, fp):
    """
    Linearly interpolate (xp, fp) to evaluate at x.
    
    Here xp must be regular (evenly spaced).

    Parameters
    ----------
    x : float, or np.array
        The x-coordinate at which to evaluate the interpolated values.

    xp : np.array(floats)
         The x-coordinates of the data points

    fp : np.array(floats)
         The y-coordinates of the data points (same length as xp)

    Returns
    -------
    y : float or np.array
        the interpolated value(s) with return type dependent on x

    Raises
    ------
    ValueError
        if xp and fp have different lengths
    """

    if len(xp) != len(fp):
        raise ValueError("xp and fp must be the same length")

    if isinstance(x, float):
        return _interpf(x, xp, fp)
    else:
        return _interpa(x, xp, fp)


@njit
def _interpf(x, xp, fp):
    """
    Linearly interpolate (xp, fp) to evaluate at x.
    
    Here xp must be regular (evenly spaced).

    Parameters
    ----------
    x : float
        The x-coordinate at which to evaluate the interpolated values.

    xp : np.array(floats)
         The x-coordinates of the data points

    fp : np.array(floats)
         The y-coordinates of the data points (same length as xp)

    Returns
    -------
    y : float
        the interpolated value

    Raises
    ------
    ValueError
        if xp and fp have different lengths
    """

    a, b, len_g = np.min(xp), np.max(xp), len(xp)
    s = (x - a) / (b - a)
    q_0 = max(min(int(s * (len_g - 1)), (len_g - 2)), 0)
    v_0 = fp[q_0]
    v_1 = fp[q_0 + 1]
    λ = s * (len_g - 1) - q_0
    y = (1 - λ) * v_0 + λ * v_1
    return y


@njit
def _interpa(x, xp, fp):
    """
    Linearly interpolate (xp, fp) to evaluate an array of x.
    
    Here xp must be regular (evenly spaced).

    Parameters
    ----------
    x : np.array(float)
        The x-coordinate's at which to evaluate the interpolated values.

    xp : np.array(float)
         The x-coordinates of the data points

    fp : np.array(float)
         The y-coordinates of the data points (same length as xp)

    Returns
    -------
    y : np.array(float)
        the interpolated values
    """

    y = np.empty_like(x)
    for idx in range(len(x)):
        y[idx] = interpf(x[idx], xp, fp)
    return y