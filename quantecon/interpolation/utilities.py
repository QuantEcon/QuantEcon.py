"""
Implements linear interpolation in up to 4 dimensions.
Based on linear interpolation code written by @albop.

"""

from numba import jit, njit
import numpy as np

@njit
def quantile(x, q):
    """
    Return, roughly, the q-th quantile of univariate data set x.

    Not exact, skips linear interpolation.  Works fine for large
    samples.
    """
    k = len(x)
    x.sort()
    return x[int(q * k)]


@njit
def lininterp_1d(grid, vals, x):
    """
    Linearly interpolate (grid, vals) to evaluate at x.
    Here grid must be regular (evenly spaced).

    Based on linear interpolation code written by @albop.

    Parameters
    ----------
    grid and vals are numpy arrays, x is a float

    Returns
    -------
    a float, the interpolated value

    """

    a, b, G = np.min(grid), np.max(grid), len(grid)

    s = (x - a) / (b - a)

    q_0 = max(min(int(s * (G - 1)), (G - 2)), 0)
    v_0 = vals[q_0]
    v_1 = vals[q_0 + 1]

    λ = s * (G - 1) - q_0

    return (1 - λ) * v_0 + λ * v_1



@njit
def lininterp_2d(x_grid, y_grid, vals, s):
    """
    Fast 2D interpolation.  Uses linear extrapolation for points outside the
    grid.

    Based on linear interpolation code written by @albop.

    Parameters
    ----------

    x_grid: np.ndarray
        grid points for x, one dimensional

    y_grid: np.ndarray
        grid points for y, one dimensional

    vals: np.ndarray
        vals[i, j] = f(x[i], y[j])

    s: np.ndarray
        2D point at which to evaluate

    """

    nx = len(x_grid)
    ny = len(y_grid)

    ax, bx = x_grid[0], x_grid[-1]
    ay, by = y_grid[0], y_grid[-1]

    s_0 = s[0]
    s_1 = s[1]

    # (s_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
    s_0 = (s_0 - ax) / (bx - ax)
    s_1 = (s_1 - ay) / (by - ay)

    # q_k : index of the interval "containing" s_k
    q_0 = max(min(int(s_0 *(nx - 1)), (nx - 2) ), 0)
    q_1 = max(min(int(s_1 *(ny - 1)), (ny - 2) ), 0)

    # lam_k : barycentric coordinate in interval k
    lam_0 = s_0 * (nx-1) - q_0
    lam_1 = s_1 * (ny-1) - q_1

    # v_ij: values on vertices of hypercube "containing" the point
    v_00 = vals[(q_0), (q_1)]
    v_01 = vals[(q_0), (q_1+1)]
    v_10 = vals[(q_0+1), (q_1)]
    v_11 = vals[(q_0+1), (q_1+1)]

    # interpolated/extrapolated value
    out = (1-lam_0) * ((1-lam_1) * (v_00) + \
                (lam_1) * (v_01)) + (lam_0) * ((1-lam_1) * (v_10) \
                + (lam_1) * (v_11))

    return out


@njit
def lininterp_3d(x_grid, y_grid, z_grid, vals, s):
    """
    Fast 3D interpolation.  Uses linear extrapolation for points outside the
    grid.  Note that the grid must be regular (i.e., evenly spaced).

    Based on linear interpolation code written by @albop.

    Parameters
    ----------

    x_grid: np.ndarray
        grid points for x, one dimensional regular grid

    y_grid: np.ndarray
        grid points for y, one dimensional regular grid

    z_grid: np.ndarray
        grid points for z, one dimensional regular grid

    vals: np.ndarray
        vals[i, j, k] = f(x[i], y[j], z[k])

    s: np.ndarray
        3D point at which to evaluate function

    """


    d = 3
    smin = (x_grid[0], y_grid[0], z_grid[0])
    smax = (x_grid[-1], y_grid[-1], z_grid[-1])

    order_0 = len(x_grid)
    order_1 = len(y_grid)
    order_2 = len(z_grid)

    # (s_1, ..., s_d) : evaluation point
    s_0 = s[0]
    s_1 = s[1]
    s_2 = s[2]

    # normalized evaluation point (in [0,1] inside the grid)
    s_0 = (s_0-smin[0])/(smax[0]-smin[0])
    s_1 = (s_1-smin[1])/(smax[1]-smin[1])
    s_2 = (s_2-smin[2])/(smax[2]-smin[2])

    # q_k : index of the interval "containing" s_k
    q_0 = max( min( int(s_0 *(order_0-1)), (order_0-2) ), 0 )
    q_1 = max( min( int(s_1 *(order_1-1)), (order_1-2) ), 0 )
    q_2 = max( min( int(s_2 *(order_2-1)), (order_2-2) ), 0 )

    # lam_k : barycentric coordinate in interval k
    lam_0 = s_0*(order_0-1) - q_0
    lam_1 = s_1*(order_1-1) - q_1
    lam_2 = s_2*(order_2-1) - q_2

    # v_ij: values on vertices of hypercube "containing" the point
    v_000 = vals[(q_0), (q_1), (q_2)]
    v_001 = vals[(q_0), (q_1), (q_2+1)]
    v_010 = vals[(q_0), (q_1+1), (q_2)]
    v_011 = vals[(q_0), (q_1+1), (q_2+1)]
    v_100 = vals[(q_0+1), (q_1), (q_2)]
    v_101 = vals[(q_0+1), (q_1), (q_2+1)]
    v_110 = vals[(q_0+1), (q_1+1), (q_2)]
    v_111 = vals[(q_0+1), (q_1+1), (q_2+1)]

    # interpolated/extrapolated value
    output = (1-lam_0)*((1-lam_1)*((1-lam_2)*(v_000) + (lam_2)*(v_001)) + (lam_1)*((1-lam_2)*(v_010) + (lam_2)*(v_011))) + (lam_0)*((1-lam_1)*((1-lam_2)*(v_100) + (lam_2)*(v_101)) + (lam_1)*((1-lam_2)*(v_110) + (lam_2)*(v_111)))
    return output


@njit
def lininterp_4d(u_grid, v_grid, w_grid, x_grid, vals, s):
    """
    Fast 4D interpolation.  Uses linear extrapolation for points outside the
    grid.  Note that the grid must be regular (i.e., evenly spaced).

    Based on linear interpolation code written by @albop.

    Parameters
    ----------

    u_grid: np.ndarray
        grid points for u, one dimensional regular grid

    v_grid: np.ndarray
        grid points for v, one dimensional regular grid

    w_grid: np.ndarray
        grid points for w, one dimensional regular grid

    x_grid: np.ndarray
        grid points for x, one dimensional regular grid

    vals: np.ndarray
        vals[i, j, k, l] = f(u[i], v[j], w[k], x[l])

    s: np.ndarray
        4D point at which to evaluate function

    """


    d = 4
    smin = (u_grid[0], v_grid[0], w_grid[0], x_grid[0])
    smax = (u_grid[-1], v_grid[-1], w_grid[-1], x_grid[-1])

    order_0 = len(u_grid)
    order_1 = len(v_grid)
    order_2 = len(w_grid)
    order_3 = len(x_grid)

    # (s_1, ..., s_d) : evaluation point
    s_0 = s[0]
    s_1 = s[1]
    s_2 = s[2]
    s_3 = s[3]

    # (s_1, ..., sn_d) : normalized evaluation point (in [0,1] inside the grid)
    s_0 = (s_0-smin[0])/(smax[0]-smin[0])
    s_1 = (s_1-smin[1])/(smax[1]-smin[1])
    s_2 = (s_2-smin[2])/(smax[2]-smin[2])
    s_3 = (s_3-smin[3])/(smax[3]-smin[3])

    # q_k : index of the interval "containing" s_k
    q_0 = max( min( int(s_0 *(order_0-1)), (order_0-2) ), 0 )
    q_1 = max( min( int(s_1 *(order_1-1)), (order_1-2) ), 0 )
    q_2 = max( min( int(s_2 *(order_2-1)), (order_2-2) ), 0 )
    q_3 = max( min( int(s_3 *(order_3-1)), (order_3-2) ), 0 )

    # lam_k : barycentric coordinate in interval k
    lam_0 = s_0*(order_0-1) - q_0
    lam_1 = s_1*(order_1-1) - q_1
    lam_2 = s_2*(order_2-1) - q_2
    lam_3 = s_3*(order_3-1) - q_3

    # v_ij: values on vertices of hypercube "containing" the point
    v_0000 = vals[(q_0), (q_1), (q_2), (q_3)]
    v_0001 = vals[(q_0), (q_1), (q_2), (q_3+1)]
    v_0010 = vals[(q_0), (q_1), (q_2+1), (q_3)]
    v_0011 = vals[(q_0), (q_1), (q_2+1), (q_3+1)]
    v_0100 = vals[(q_0), (q_1+1), (q_2), (q_3)]
    v_0101 = vals[(q_0), (q_1+1), (q_2), (q_3+1)]
    v_0110 = vals[(q_0), (q_1+1), (q_2+1), (q_3)]
    v_0111 = vals[(q_0), (q_1+1), (q_2+1), (q_3+1)]
    v_1000 = vals[(q_0+1), (q_1), (q_2), (q_3)]
    v_1001 = vals[(q_0+1), (q_1), (q_2), (q_3+1)]
    v_1010 = vals[(q_0+1), (q_1), (q_2+1), (q_3)]
    v_1011 = vals[(q_0+1), (q_1), (q_2+1), (q_3+1)]
    v_1100 = vals[(q_0+1), (q_1+1), (q_2), (q_3)]
    v_1101 = vals[(q_0+1), (q_1+1), (q_2), (q_3+1)]
    v_1110 = vals[(q_0+1), (q_1+1), (q_2+1), (q_3)]
    v_1111 = vals[(q_0+1), (q_1+1), (q_2+1), (q_3+1)]

    # interpolated/extrapolated value
    output = (1-lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_0000) + (lam_3)*(v_0001)) + (lam_2)*((1-lam_3)*(v_0010) + (lam_3)*(v_0011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_0100) + (lam_3)*(v_0101)) + (lam_2)*((1-lam_3)*(v_0110) + (lam_3)*(v_0111)))) + (lam_0)*((1-lam_1)*((1-lam_2)*((1-lam_3)*(v_1000) + (lam_3)*(v_1001)) + (lam_2)*((1-lam_3)*(v_1010) + (lam_3)*(v_1011))) + (lam_1)*((1-lam_2)*((1-lam_3)*(v_1100) + (lam_3)*(v_1101)) + (lam_2)*((1-lam_3)*(v_1110) + (lam_3)*(v_1111))))

    return output
