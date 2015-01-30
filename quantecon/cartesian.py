"""
Filename: cartesian.py

Authors: Pablo Winant

Implements cartesian products and regular cartesian grids.
"""

import numpy
from numba import njit


def cartesian(nodes, order='C'):
    '''Cartesian product of a list of arrays

    Parameters:
    -----------
    nodes: (list of 1d-arrays)
    order: ('C' or 'F') order in which the product is enumerated

    Returns:
    --------
    out: (2d-array) each line corresponds to one point of the product space
    '''

    nodes = [numpy.array(e) for e in nodes]
    shapes = [e.shape[0] for e in nodes]

    dtype = nodes[0].dtype

    n = len(nodes)
    l = numpy.prod(shapes)
    out = numpy.zeros((l, n), dtype=dtype)

    if order == 'C':
        repetitions = numpy.cumprod([1] + shapes[:-1])
    else:
        shapes.reverse()
        sh = [1] + shapes[:-1]
        repetitions = numpy.cumprod(sh)
        repetitions = repetitions.tolist()
        repetitions.reverse()

    for i in range(n):
        _repeat_1d(nodes[i], repetitions[i], out[:, i])

    return out


def mlinspace(a, b, nums, order='C'):
    '''Constructs a regular cartesian grid

    Parameters:
    -----------
    a: (1d-array) lower bounds in each dimension
    b: (1d-array) upper bounds in each dimension
    nums: (1d-array) number of nodes along each dimension
    order: ('C' or 'F') order in which the product is enumerated

    Returns:
    --------
    out: (2d-array) each line corresponds to one point of the product space
    '''

    a = numpy.array(a, dtype='float64')
    b = numpy.array(b, dtype='float64')
    nums = numpy.array(nums, dtype='int64')
    nodes = [numpy.linspace(a[i], b[i], nums[i]) for i in range(len(nums))]

    return cartesian(nodes, order=order)


@njit
def _repeat_1d(x, K, out):
    '''
    Repeats each element of a vector many times and repeats the
    whole result many times

    Parameters
    ----------

    x : array_like(Any, ndim=1)
        The vector to be repeated
    K : scalar(int)
        The number of times each element of x
        is repeated (inner iterations)
    out : array_like(Any, ndim=1)
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
