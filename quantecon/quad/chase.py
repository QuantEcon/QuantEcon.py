from __future__ import division
import numpy as np
from math import exp, factorial, log
from scipy.special import gammaln


def _qnwbeta1(n, a=1, b=1):
    """
    Insert docs.  Default is a=b=1 which is just a uniform distribution

    NOTE: Add multidimensional support with wrapper function around
    this one.

    NOTE: For now I am just following compecon; would be much better to
    find a different way since I don't know what they are doing.

    Parameters
    ----------
    n : scalar : int
        The number of quadrature points

    a : scalar : float
        First Beta distribution parameter (default value is 1)

    b : scalar : float
        Second Beta distribution parameter (default value is 1)

    Returns
    -------
    nodes : array(ndim=1) : float
        The quadrature points

    weights : array(ndim=1) : float
        The quadrature weights that correspond to nodes
    """
    # We subtract one and write a + 1 where we actually want a, and a
    # where we want a - 1
    a = a-1
    b = b-1

    maxiter=25

    # Allocate empty space
    nodes = np.zeros(n)
    weights = np.zeros(n)

    # Find "reasonable" starting values.  Why these numbers?
    for i in range(n):
        if i==0:
            an = a/n
            bn = b/n
            r1 = (1+a) * (2.78/(4+n*n) + .768*an/n)
            r2 = 1 + 1.48*an + .96*bn + .452*an*an + .83*an*bn
            z = 1 - r1/r2
        elif i==1:
            r1 = (4.1+a) / ((1+a)*(1+0.156*a))
            r2 = 1 + 0.06 * (n-8) * (1+0.12*a)/n
            r3 = 1 + 0.012*b * (1+0.25*abs(a))/n
            z = z - (1-z) * r1 * r2 * r3
        elif i==2:
            r1 = (1.67+0.28*a)/(1+0.37*a)
            r2 = 1+0.22*(n-8)/n
            r3 = 1+8*b/((6.28+b)*n*n)
            z = z-(nodes[0]-z)*r1*r2*r3
        elif i==n-2:
            r1 = (1+0.235*b)/(0.766+0.119*b)
            r2 = 1/(1+0.639*(n-4)/(1+0.71*(n-4)))
            r3 = 1/(1+20*a/((7.5+a)*n*n))
            z = z+(z-nodes[-4])*r1*r2*r3
        elif i==n-1:
            r1 = (1+0.37*b) / (1.67+0.28*b)
            r2 = 1 / (1+0.22*(n-8)/n)
            r3 = 1 / (1+8*a/((6.28+a)*n*n))
            z = z+(z-nodes[-3])*r1*r2*r3
        else:
            z = 3*nodes[i-1] - 3*nodes[i-2] + nodes[i-3]

        ab = a+b

        # Root finding
        its = 0
        z1 = -100
        while abs(z - z1) > 1e-10 and its < maxiter:
            temp = 2 + ab
            p1 = (a-b + temp*z)/2
            p2 = 1

            for j in range(2, n+1):
                p3 = p2
                p2 = p1
                temp = 2*j + ab
                aa = 2*j * (j+ab)*(temp-2)
                bb = (temp-1) * (a*a - b*b + temp*(temp-2) * z)
                c = 2 * (j - 1 + a) * (j - 1 + b) * temp
                p1 = (bb*p2 - c*p3)/aa

            pp = (n*(a-b-temp*z) * p1 + 2*(n+a)*(n+b)*p2)/(temp*(1 - z*z))
            z1 = z
            z = z1 - p1/pp

            if abs(z - z1) < 1e-12:
                break

            its += 1

        if its==maxiter:
            raise ValueError("Max Iteration reached.  Failed to converge")

        nodes[i] = z
        weights[i] = temp/(pp*p2)

    nodes = (1-nodes)/2
    weights = weights * exp(gammaln(a+n) + gammaln(b+n)
                            - gammaln(n+1) - gammaln(n+ab+1))
    weights = weights / (2*exp(gammaln(a+1) + gammaln(b+1)
                         - gammaln(ab+2)))

    return nodes, weights


def qnwbeta(n, a=1, b=1):
    """
    multi dimensional case
    """
    if type(n) == int:
        return _qnwbeta(n, a, b)

    else:
        d = n.size

        nodes = []
        weights = []

        for i in range(d):
            temp_node, temp_w = _qnwbeta1(n[i], a[i], b[i])

            nodes.append(temp_node)
            weights.append(temp_weights)

        weights = ckron(*weights[::-1])
        nodes = gridmake(*nodes)

        return nodes, weights


def _qnwgamma1(n, a=0):
    """
    Insert docs.  Default is a=0

    NOTE: Add multidimensional support with wrapper function around
    this one.

    NOTE: For now I am just following compecon; would be much better to
    find a different way since I don't know what they are doing.

    Parameters
    ----------
    n : scalar : int
        The number of quadrature points

    a : scalar : float
        Gamma distribution parameter

    Returns
    -------
    nodes : array(ndim=1) : float
        The quadrature points

    weights : array(ndim=1) : float
        The quadrature weights that correspond to nodes
    """
    if a>0:
        a = a-1
    maxit = 10

    factor = -exp(gammaln(a+n) - gammaln(n) - gammaln(a+1))
    nodes = np.zeros(n)
    weights = np.zeros(n)

    # Create nodes
    for i in range(n):
        # Reasonable starting values
        if i==0:
            z = (1+a) * (3+0.92*a) / (1 + 2.4*n + 1.8*a)
        elif i==1:
            z = z + (15 + 6.25*a) / (1 + 0.9*a + 2.5*n)
        else:
          j = i-1
          z = z + ((1 + 2.55*j) / (1.9*j) + 1.26*j*a / (1 + 3.5*j)) * \
          (z - nodes[j]) / (1 + 0.3*a)

        print(i, z)

        # root finding iterations
        its = 0
        z1 = -10
        while abs(z - z1)>1e-10 and its < maxit:
            p1 = 1
            p2 = 0
            for j in range(1, n+1):
                p3 = p2
                p2 = p1
                p1 = ((2*j - 1 + a - z)*p2 - (j - 1 + a)*p3) / j

            pp = (n*p1 - (n+a)*p2) / z
            z1 = z
            z = z1 - p1/pp
            its += 1

        if its==maxit:
            raise ValueError('Failure to converge')

        nodes[i] = z
        weights[i] = factor / (pp*n*p2)

    return nodes, weights
