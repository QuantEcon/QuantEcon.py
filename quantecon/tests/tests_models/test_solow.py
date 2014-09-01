"""
Tests for quantecon.models.solow

@author : David R. Pugh
@date : 2014-08-18

"""
from __future__ import division

import sympy as sp

from quantecon.models import solow

# model variables
A, K, L = sp.var('A, K, L')

# production parameters
alpha, sigma = sp.var('alpha, sigma')

# CES production function
rho = (sigma - 1) / sigma
Y = (alpha * K**rho + (1 - alpha) * (A * L)**rho)**(1 / rho)

# model parameters
test_params = {'g': 0.02, 'n': 0.02, 's': 0.15, 'delta': 0.04, 'alpha': 0.33,
               'sigma': 0.95}

# model object
model = solow.Model(output=Y, params=test_params)
