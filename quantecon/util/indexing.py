"""
Indexing Utilities

Utilities
---------
index_dict

"""
import numpy as np
from numba import jit, types
from numba.typed import Dict


@jit(nopython=True, cache=True)
def _fill_1d(dd, arr):
    for i in range(arr.shape[0]):
        dd[arr[i]] = i


# Cache of jitted fill functions for 2-dimensional values arrays, keyed
# by the number of columns (= the length of the tuple keys), which must
# be a compile-time constant for tuple construction under Numba
_fill_nd_cache = {}


def _make_fill_nd(d):
    fill = _fill_nd_cache.get(d)
    if fill is None:
        elems = ", ".join(f"arr[i, {k}]" for k in range(d))
        code = (
            "def _fill(dd, arr):\n"
            "    for i in range(arr.shape[0]):\n"
            f"        dd[({elems},)] = i\n"
        )
        ns = {}
        exec(code, ns)
        fill = jit(nopython=True)(ns["_fill"])
        _fill_nd_cache[d] = fill
    return fill


def _diagnose_nonfinite(arr):
    isfinite = np.isfinite(arr)
    if arr.ndim == 2:
        isfinite = isfinite.all(axis=1)
    i = int(np.argmin(isfinite))
    v = arr[i] if arr.ndim == 1 else tuple(arr[i])
    raise ValueError(f"values contains non-finite value {v} at index {i}")


def _diagnose_negative_zero(arr):
    is_negzero = np.signbit(arr) & (arr == 0)
    if arr.ndim == 2:
        is_negzero = is_negzero.any(axis=1)
    i = int(np.argmax(is_negzero))
    raise ValueError(
        f"values contains -0.0 at index {i}; use 0.0 instead"
    )


def _diagnose_duplicates(arr):
    seen = {}
    for i in range(arr.shape[0]):
        key = arr[i] if arr.ndim == 1 else tuple(arr[i])
        if key in seen:
            raise ValueError(
                f"duplicate value {key} at indices {seen[key]} and {i}"
            )
        seen[key] = i


def index_dict(values):
    """
    Build a Numba typed dictionary mapping each value in `values` to its
    index.

    Parameters
    ----------
    values : array_like
        Array of unique values, of shape (n,) for scalar values or
        (n, d) for vector values. dtype must be of float or integer
        kind (the values are stored as float64 or int64 keys); values
        must be finite.

    Returns
    -------
    numba.typed.Dict
        Dict mapping value -> index, with int64 indices. For an input
        of shape (n,), keys are scalars (float64 or int64, following
        the dtype kind of `values`); for shape (n, d), keys are
        d-tuples thereof. Usable both from the interpreter and inside
        Numba-jitted functions, and passable to jitted functions as an
        argument.

    Raises
    ------
    ValueError
        If `values` contains duplicate, non-finite, or negative-zero
        values, or is not 1- or 2-dimensional.

    Notes
    -----
    Lookup is by exact equality: query keys must be values drawn from
    `values` itself (or arithmetic that reproduces them bitwise). For
    approximate location on a sorted grid, use `np.searchsorted`
    instead.

    Numba typed dicts hash floats by bit pattern, so `-0.0` and `0.0`
    are distinct keys (unlike in a Python dict). `values` therefore
    must not contain `-0.0`; a query key that may have been computed as
    `-0.0` can be normalized by adding `0.0` to it.

    Examples
    --------
    >>> d = index_dict(np.array([0.1, 1.0]))
    >>> d[1.0]
    1
    >>> d2 = index_dict(np.array([[0., 0.1], [0., 1.], [0.5, 0.1]]))
    >>> d2[(0.5, 0.1)]
    2

    """
    arr = np.asarray(values)
    if arr.ndim not in (1, 2):
        raise ValueError("values must be 1- or 2-dimensional")
    if arr.ndim == 2 and arr.shape[1] == 0:
        raise ValueError("values must have at least one column")

    if arr.dtype.kind == 'f':
        arr = arr.astype(np.float64, copy=False)
        scalar_type = types.float64
        if not np.isfinite(arr).all():
            _diagnose_nonfinite(arr)
        if np.any(np.signbit(arr) & (arr == 0)):
            _diagnose_negative_zero(arr)
    elif arr.dtype.kind in 'iu':
        arr = arr.astype(np.int64, copy=False)
        scalar_type = types.int64
    else:
        raise ValueError(f"unsupported dtype: {arr.dtype}")

    if arr.ndim == 1:
        key_type = scalar_type
        fill = _fill_1d
    else:
        key_type = types.UniTuple(scalar_type, arr.shape[1])
        fill = _make_fill_nd(arr.shape[1])

    dd = Dict.empty(key_type, types.int64)
    fill(dd, arr)

    if len(dd) != arr.shape[0]:
        _diagnose_duplicates(arr)

    return dd
