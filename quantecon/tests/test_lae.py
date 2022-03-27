"""
Tests for lae.py

TODO: write (economically) meaningful tests for this module

"""
import numpy as np
from numpy.testing import assert_
from scipy.stats import lognorm
from quantecon import LAE

# copied from the lae lecture
s = 0.2
delta = 0.1
a_sigma = 0.4       # A = exp(B) where B ~ N(0, a_sigma)
alpha = 0.4         # We set f(k) = k**alpha
phi = lognorm(a_sigma)


def p(x, y):
    d = s * x**alpha
    return phi.pdf((y - (1 - delta) * x) / d) / d

# other data
n_a, n_b, n_y = 50, (5, 5), 20
a = np.random.rand(n_a) + 0.01
b = np.random.rand(*n_b) + 0.01

y = np.linspace(0, 10, 20)

lae_a = LAE(p, a)
lae_b = LAE(p, b)


def test_x_flattened():
    "lae: is x flattened and reshaped"
    # should have a trailing singleton dimension
    assert_(lae_b.X.shape[-1] == 1)
    assert_(lae_a.X.shape[-1] == 1)


def test_x_2d():
    "lae: is x 2d"
    assert_(lae_a.X.ndim == 2)
    assert_(lae_b.X.ndim == 2)


def test_call_shapes():
    "lae: shape of call to lae"
    assert_(lae_a(y).shape == (n_y,))
    assert_(lae_b(y).shape == (n_y,))
