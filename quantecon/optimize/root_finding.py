import numpy as np
from numba import njit
from collections import namedtuple

__all__ = ['newton', 'newton_halley', 'newton_secant', 'bisect', 'brentq']

_ECONVERGED = 0
_ECONVERR = -1

_iter = 100
_xtol = 2e-12
_rtol = 4*np.finfo(float).eps

results = namedtuple('results', 'root function_calls iterations converged')


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
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    tol : float, optional(default=1.48e-8)
        The allowable error of the zero value.
    maxiter : int, optional(default=50)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge

    Returns
    -------
    results : namedtuple
        A namedtuple containing the following items:
        ::

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
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    tol : float, optional(default=1.48e-8)
        The allowable error of the zero value.
    maxiter : int, optional(default=50)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge

    Returns
    -------
    results : namedtuple
        A namedtuple containing the following items:
        ::

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
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    tol : float, optional(default=1.48e-8)
        The allowable error of the zero value.
    maxiter : int, optional(default=50)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results : namedtuple
        A namedtuple containing the following items:
        ::

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


@njit
def _bisect_interval(a, b, fa, fb):
    """Conditional checks for intervals in methods involving bisection"""
    if fa*fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    root = 0.0
    status = _ECONVERR

    # Root found at either end of [a,b]
    if fa == 0:
        root = a
        status = _ECONVERGED
    if fb == 0:
        root = b
        status = _ECONVERGED

    return root, status


@njit
def bisect(f, a, b, args=(), xtol=_xtol,
           rtol=_rtol, maxiter=_iter, disp=True):
    """
    Find root of a function within an interval adapted from Scipy's bisect.

    Basic bisection routine to find a zero of the function `f` between the
    arguments `a` and `b`. `f(a)` and `f(b)` cannot have the same signs.

    `f` must be jitted via numba.

    Parameters
    ----------
    f : jitted and callable
        Python function returning a number.  `f` must be continuous.
    a : number
        One end of the bracketing interval [a,b].
    b : number
        The other end of the bracketing interval [a,b].
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    xtol : number, optional(default=2e-12)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be nonnegative.
    rtol : number, optional(default=4*np.finfo(float).eps)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : number, optional(default=100)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results : namedtuple

    """

    if xtol <= 0:
        raise ValueError("xtol is too small (<= 0)")

    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float
    xa = a * 1.0
    xb = b * 1.0

    fa = f(xa, *args)
    fb = f(xb, *args)
    funcalls = 2
    root, status = _bisect_interval(xa, xb, fa, fb)

    # Check for sign error and early termination
    if status == _ECONVERGED:
        itr = 0
    else:
        # Perform bisection
        dm = xb - xa
        for itr in range(maxiter):
            dm *= 0.5
            xm = xa + dm
            fm = f(xm, *args)
            funcalls += 1

            if fm * fa >= 0:
                xa = xm

            if fm == 0 or abs(dm) < xtol + rtol * abs(xm):
                root = xm
                status = _ECONVERGED
                itr += 1
                break

    if disp and status == _ECONVERR:
        raise RuntimeError("Failed to converge")

    return _results((root, funcalls, itr, status))


@njit
def brentq(f, a, b, args=(), xtol=_xtol,
           rtol=_rtol, maxiter=_iter, disp=True):
    """
    Find a root of a function in a bracketing interval using Brent's method
    adapted from Scipy's brentq.

    Uses the classic Brent's method to find a zero of the function `f` on
    the sign changing interval [a , b].

    `f` must be jitted via numba.

    Parameters
    ----------
    f : jitted and callable
        Python function returning a number.  `f` must be continuous.
    a : number
        One end of the bracketing interval [a,b].
    b : number
        The other end of the bracketing interval [a,b].
    args : tuple, optional(default=())
        Extra arguments to be used in the function call.
    xtol : number, optional(default=2e-12)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root. The
        parameter must be nonnegative.
    rtol : number, optional(default=4*np.finfo(float).eps)
        The computed root ``x0`` will satisfy ``np.allclose(x, x0,
        atol=xtol, rtol=rtol)``, where ``x`` is the exact root.
    maxiter : number, optional(default=100)
        Maximum number of iterations.
    disp : bool, optional(default=True)
        If True, raise a RuntimeError if the algorithm didn't converge.

    Returns
    -------
    results : namedtuple

    """
    if xtol <= 0:
        raise ValueError("xtol is too small (<= 0)")
    if maxiter < 1:
        raise ValueError("maxiter must be greater than 0")

    # Convert to float
    xpre = a * 1.0
    xcur = b * 1.0

    fpre = f(xpre, *args)
    fcur = f(xcur, *args)
    funcalls = 2

    root, status = _bisect_interval(xpre, xcur, fpre, fcur)

    # Check for sign error and early termination
    if status == _ECONVERGED:
        itr = 0
    else:
        # Perform Brent's method
        for itr in range(maxiter):

            if fpre * fcur < 0:
                xblk = xpre
                fblk = fpre
                spre = scur = xcur - xpre
            if abs(fblk) < abs(fcur):
                xpre = xcur
                xcur = xblk
                xblk = xpre

                fpre = fcur
                fcur = fblk
                fblk = fpre

            delta = (xtol + rtol * abs(xcur)) / 2
            sbis = (xblk - xcur) / 2

            # Root found
            if fcur == 0 or abs(sbis) < delta:
                status = _ECONVERGED
                root = xcur
                itr += 1
                break

            if abs(spre) > delta and abs(fcur) < abs(fpre):
                if xpre == xblk:
                    # interpolate
                    stry = -fcur * (xcur - xpre) / (fcur - fpre)
                else:
                    # extrapolate
                    dpre = (fpre - fcur) / (xpre - xcur)
                    dblk = (fblk - fcur) / (xblk - xcur)
                    stry = -fcur * (fblk * dblk - fpre * dpre) / \
                        (dblk * dpre * (fblk - fpre))

                if (2 * abs(stry) < min(abs(spre), 3 * abs(sbis) - delta)):
                    # good short step
                    spre = scur
                    scur = stry
                else:
                    # bisect
                    spre = sbis
                    scur = sbis
            else:
                # bisect
                spre = sbis
                scur = sbis

            xpre = xcur
            fpre = fcur
            if (abs(scur) > delta):
                xcur += scur
            else:
                xcur += (delta if sbis > 0 else -delta)
            fcur = f(xcur, *args)
            funcalls += 1

    if disp and status == _ECONVERR:
        raise RuntimeError("Failed to converge")

    return _results((root, funcalls, itr, status))
