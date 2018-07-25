import numpy as np
from numba import jit, njit
from collections import namedtuple

__all__ = ['newton', 'newton_halley', 'newton_secant']

_ECONVERGED = 0
_ECONVERR = -1

results = namedtuple('results', 
                      ('root function_calls iterations converged'))

@njit
def _results(r):
    r"""Select from a tuple of(root, funccalls, iterations, flag)"""
    x, funcalls, iterations, flag = r
    return results(x, funcalls, iterations, flag == 0)

@njit
def newton(func, x0, fprime, args=(), tol=1.48e-8, maxiter=50,
           disp=True):
    """
    Find a zero from the Newton-Raphson method using the jitted version of 
    Scipy's newton for scalars. Note that this does not provide an alternative 
    method such as secant. Thus, it is important that `fprime` can be provided.

    Note that `func` and `fprime` must be jitted via Numba.
    They are recommended to be `njit` for performance.

    Parameters
    ----------
    func : callable and jitted
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    fprime : callable and jitted
        The derivative of the function (when available and convenient).
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations.
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge

    Returns
    -------
    results : namedtuple
        root - Estimated location where function is zero.
        function_calls - Number of times the function was called.
        iterations - Number of iterations needed to find the root.
        converged - True if the routine converged
    """

    if tol <= 0:
        raise ValueError("tol is too small <= 0")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    status = _ECONVERR

    # Newton-Raphson method
    for itr in range(maxiter):
        # first evaluate fval
        fval = func(p0, *args)
        funcalls += 1
        # If fval is 0, a root has been found, then terminate
        if fval == 0:
            status = _ECONVERGED
            p = p0
            itr -= 1
            break
        fder = fprime(p0, *args)
        funcalls += 1
        # derivative is zero, not converged
        if fder == 0:
            p = p0
            break
        newton_step = fval / fder
        # Newton step
        p = p0 - newton_step   
        if abs(p - p0) < tol:
            status = _ECONVERGED
            break
        p0 = p
    
    if disp and status == _ECONVERR:
        msg = "Failed to converge"
        raise RuntimeError(msg)

    return _results((p, funcalls, itr + 1, status))

@njit
def newton_halley(func, x0, fprime, fprime2, args=(), tol=1.48e-8,
                  maxiter=50, disp=True):
    """
    Find a zero from Halley's method using the jitted version of
    Scipy's.

    `func`, `fprime`, `fprime2` must be jitted via Numba.

    Parameters
    ----------
    func : callable and jitted
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    fprime : callable and jitted
        The derivative of the function (when available and convenient).
    fprime2 : callable and jitted
        The second order derivative of the function
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations.
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge

    Returns
    -------
    results : namedtuple
        root - Estimated location where function is zero.
        function_calls - Number of times the function was called.
        iterations - Number of iterations needed to find the root.
        converged - True if the routine converged
    """

    if tol <= 0:
        raise ValueError("tol is too small <= 0")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    status = _ECONVERR

    # Halley Method
    for itr in range(maxiter):
        # first evaluate fval
        fval = func(p0, *args)
        funcalls += 1
        # If fval is 0, a root has been found, then terminate
        if fval == 0:
            status = _ECONVERGED
            p = p0
            itr -= 1
            break
        fder = fprime(p0, *args)
        funcalls += 1
        # derivative is zero, not converged
        if fder == 0:
            p = p0
            break
        newton_step = fval / fder
        # Halley's variant
        fder2 = fprime2(p0, *args)
        p = p0 - newton_step / (1.0 - 0.5 * newton_step * fder2 / fder)
        if abs(p - p0) < tol:
            status = _ECONVERGED
            break
        p0 = p

    if disp and status == _ECONVERR:
        msg = "Failed to converge"
        raise RuntimeError(msg)

    return _results((p, funcalls, itr + 1, status))

@njit
def newton_secant(func, x0, args=(), tol=1.48e-8, maxiter=50,
                  disp=True):
    """
    Find a zero from the secant method using the jitted version of 
    Scipy's secant method.
    
    Note that `func` must be jitted via Numba.
    
    Parameters
    ----------
    func : callable and jitted
        The function whose zero is wanted. It must be a function of a
        single variable of the form f(x,a,b,c...), where a,b,c... are extra
        arguments that can be passed in the `args` parameter.
    x0 : float
        An initial estimate of the zero that should be somewhere near the
        actual zero.
    args : tuple, optional
        Extra arguments to be used in the function call.
    tol : float, optional
        The allowable error of the zero value.
    maxiter : int, optional
        Maximum number of iterations.
    disp : bool, optional
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results : namedtuple
        root - Estimated location where function is zero.
        function_calls - Number of times the function was called.
        iterations - Number of iterations needed to find the root.
        converged - True if the routine converged
    """
    
    if tol <= 0:
        raise ValueError("tol is too small <= 0")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")
    
    # Convert to float (don't use float(x0); this works also for complex x0)
    p0 = 1.0 * x0
    funcalls = 0
    status = _ECONVERR
    
    # Secant method
    if x0 >= 0:
        p1 = x0 * (1 + 1e-4) + 1e-4
    else:
        p1 = x0 * (1 + 1e-4) - 1e-4
        q0 = func(p0, *args)
    funcalls += 1
    q1 = func(p1, *args)
    funcalls += 1
    for itr in range(maxiter):
        if q1 == q0:
            p = (p1 + p0) / 2.0
            status = _ECONVERGED
            break
        else:
            p = p1 - q1 * (p1 - p0) / (q1 - q0)
        if np.abs(p - p1) < tol:
            status = _ECONVERGED
            break
        p0 = p1
        q0 = q1
        p1 = p
        q1 = func(p1, *args)
        funcalls += 1
        
    if disp and status == _ECONVERR:
        msg = "Failed to converge"
        raise RuntimeError(msg)

    return _results((p, funcalls, itr + 1, status))