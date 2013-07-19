## Filename: mc_test.py
## Author: John Stachurski


import numpy as np
from mc_stationary1 import stationary1 
from mc_stationary3 import stationary3

p = np.loadtxt('matrix_dat.txt')

def test_estimator(q, f, replications=100):
    """
    The estimator f returns an estimate of q.  Draw n=replications 
    observations and return average d1 distance

    Parameters:

        * q is a NumPy array representing the exact distribution
        * f is a function that, when called, returns an estimate of q
            as a NumPy array
        * replications is a positive integer giving the sample size

    Returns: A float
    """
    results = np.empty(replications)
    for i in range(replications):
        results[i] = np.sum(np.abs(f() - q))
    return results.mean()


if __name__ == '__main__':

    q = stationary1(p)  # Exact stationary distribution

    print "Standard MC, average distance:"
    print test_estimator(q, lambda: stationary3(p))

    print "Look-ahead MC, average distance:"
    print test_estimator(q, lambda: stationary3(p, lae=True))






