def bisect(f, a, b, tol=10e-5):
    """
    Implements the bisection root finding algorithm, assuming that f is a
    real-valued function on [a, b] satisfying f(a) < 0 < f(b).
    """
    lower, upper = a, b
    while upper - lower > tol:
        middle = 0.5 * (upper + lower)
        if f(middle) > 0:  # Implies root is between lower and middle
            lower, upper = lower, middle
        else:              # Implies root is between middle and upper
            lower, upper = middle, upper
    return 0.5 * (upper + lower)


