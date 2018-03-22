"""
Implements cartesian products and regular cartesian grids, and provides
a function that constructs a grid for a simplex as well as one that
determines the index of a point in the simplex.

"""
import numpy as np
import scipy.special
from numba import jit, njit
from .util.numba import comb_jit


def cartesian(nodes, order='C'):
    '''
    Cartesian product of a list of arrays

    Parameters
    ----------
    nodes : list(array_like(ndim=1))

    order : str, optional(default='C')
        ('C' or 'F') order in which the product is enumerated

    Returns
    -------
    out : ndarray(ndim=2)
        each line corresponds to one point of the product space
    '''

    nodes = [np.array(e) for e in nodes]
    shapes = [e.shape[0] for e in nodes]

    dtype = nodes[0].dtype

    n = len(nodes)
    l = np.prod(shapes)
    out = np.zeros((l, n), dtype=dtype)

    if order == 'C':
        repetitions = np.cumprod([1] + shapes[:-1])
    else:
        shapes.reverse()
        sh = [1] + shapes[:-1]
        repetitions = np.cumprod(sh)
        repetitions = repetitions.tolist()
        repetitions.reverse()

    for i in range(n):
        _repeat_1d(nodes[i], repetitions[i], out[:, i])

    return out


def mlinspace(a, b, nums, order='C'):
    '''
    Constructs a regular cartesian grid

    Parameters
    ----------
    a : array_like(ndim=1)
        lower bounds in each dimension

    b : array_like(ndim=1)
        upper bounds in each dimension

    nums : array_like(ndim=1)
        number of nodes along each dimension

    order : str, optional(default='C')
        ('C' or 'F') order in which the product is enumerated

    Returns
    -------
    out : ndarray(ndim=2)
        each line corresponds to one point of the product space
    '''

    a = np.array(a, dtype='float64')
    b = np.array(b, dtype='float64')
    nums = np.array(nums, dtype='int64')
    nodes = [np.linspace(a[i], b[i], nums[i]) for i in range(len(nums))]

    return cartesian(nodes, order=order)


@njit
def _repeat_1d(x, K, out):
    '''
    Repeats each element of a vector many times and repeats the whole
    result many times

    Parameters
    ----------
    x : ndarray(ndim=1)
        vector to be repeated

    K : scalar(int)
        number of times each element of x is repeated (inner iterations)

    out : ndarray(ndim=1)
        placeholder for the result

    Returns
    -------
    None
    '''

    N = x.shape[0]
    L = out.shape[0] // (K*N)  # number of outer iterations
    # K                        # number of inner iterations

    # the result out should enumerate in C-order the elements
    # of a 3-dimensional array T of dimensions (K,N,L)
    # such that for all k,n,l, we have T[k,n,l] == x[n]

    for n in range(N):
        val = x[n]
        for k in range(K):
            for l in range(L):
                ind = k*N*L + n*L + l
                out[ind] = val


_msg_max_size_exceeded = 'Maximum allowed size exceeded'


@jit(nopython=True, cache=True)
def simplex_grid(m, n):
    r"""
    Construct an array consisting of the integer points in the
    (m-1)-dimensional simplex :math:`\{x \mid x_0 + \cdots + x_{m-1} = n
    \}`, or equivalently, the m-part compositions of n, which are listed
    in lexicographic order. The total number of the points (hence the
    length of the output array) is L = (n+m-1)!/(n!*(m-1)!) (i.e.,
    (n+m-1) choose (m-1)).

    Parameters
    ----------
    m : scalar(int)
        Dimension of each point. Must be a positive integer.

    n : scalar(int)
        Number which the coordinates of each point sum to. Must be a
        nonnegative integer.

    Returns
    -------
    out : ndarray(int, ndim=2)
        Array of shape (L, m) containing the integer points in the
        simplex, aligned in lexicographic order.

    Notes
    -----
    A grid of the (m-1)-dimensional *unit* simplex with n subdivisions
    along each dimension can be obtained by `simplex_grid(m, n) / n`.

    Examples
    --------
    >>> simplex_grid(3, 4)
    array([[0, 0, 4],
           [0, 1, 3],
           [0, 2, 2],
           [0, 3, 1],
           [0, 4, 0],
           [1, 0, 3],
           [1, 1, 2],
           [1, 2, 1],
           [1, 3, 0],
           [2, 0, 2],
           [2, 1, 1],
           [2, 2, 0],
           [3, 0, 1],
           [3, 1, 0],
           [4, 0, 0]])

    >>> simplex_grid(3, 4) / 4
    array([[ 0.  ,  0.  ,  1.  ],
           [ 0.  ,  0.25,  0.75],
           [ 0.  ,  0.5 ,  0.5 ],
           [ 0.  ,  0.75,  0.25],
           [ 0.  ,  1.  ,  0.  ],
           [ 0.25,  0.  ,  0.75],
           [ 0.25,  0.25,  0.5 ],
           [ 0.25,  0.5 ,  0.25],
           [ 0.25,  0.75,  0.  ],
           [ 0.5 ,  0.  ,  0.5 ],
           [ 0.5 ,  0.25,  0.25],
           [ 0.5 ,  0.5 ,  0.  ],
           [ 0.75,  0.  ,  0.25],
           [ 0.75,  0.25,  0.  ],
           [ 1.  ,  0.  ,  0.  ]])

    References
    ----------
    A. Nijenhuis and H. S. Wilf, Combinatorial Algorithms, Chapter 5,
    Academic Press, 1978.

    """
    L = num_compositions_jit(m, n)
    if L == 0:  # Overflow occured
    	raise ValueError(_msg_max_size_exceeded)
    out = np.empty((L, m), dtype=np.int_)

    x = np.zeros(m, dtype=np.int_)
    x[m-1] = n

    for j in range(m):
        out[0, j] = x[j]

    h = m

    for i in range(1, L):
        h -= 1

        val = x[h]
        x[h] = 0
        x[m-1] = val - 1
        x[h-1] += 1

        for j in range(m):
            out[i, j] = x[j]

        if val != 1:
            h = m

    return out


def simplex_index(x, m, n):
    r"""
    Return the index of the point x in the lexicographic order of the
    integer points of the (m-1)-dimensional simplex :math:`\{x \mid x_0
    + \cdots + x_{m-1} = n\}`.

    Parameters
    ----------
    x : array_like(int, ndim=1)
        Integer point in the simplex, i.e., an array of m nonnegative
        itegers that sum to n.

    m : scalar(int)
        Dimension of each point. Must be a positive integer.

    n : scalar(int)
        Number which the coordinates of each point sum to. Must be a
        nonnegative integer.

    Returns
    -------
    idx : scalar(int)
        Index of x.

    """
    if m == 1:
        return 0

    decumsum = np.cumsum(x[-1:0:-1])[::-1]
    idx = num_compositions(m, n) - 1
    for i in range(m-1):
        if decumsum[i] == 0:
            break
        idx -= num_compositions(m-i, decumsum[i]-1)
    return idx


def num_compositions(m, n):
    """
    The total number of m-part compositions of n, which is equal to
    (n+m-1) choose (m-1).

    Parameters
    ----------
    m : scalar(int)
        Number of parts of composition.

    n : scalar(int)
        Integer to decompose.

    Returns
    -------
    scalar(int)
        Total number of m-part compositions of n.

    """
    # docs.scipy.org/doc/scipy/reference/generated/scipy.special.comb.html
    return scipy.special.comb(n+m-1, m-1, exact=True)


@jit(nopython=True, cache=True)
def num_compositions_jit(m, n):
    """
    Numba jit version of `num_compositions`. Return `0` if the outcome
    exceeds the maximum value of `np.intp`.

    """
    return comb_jit(n+m-1, m-1)
