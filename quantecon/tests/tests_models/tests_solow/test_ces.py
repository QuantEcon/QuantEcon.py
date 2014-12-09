"""
Test suite for ces.py module.

@author : David R. Pugh
@date : 2014-12-08

"""
import nose

import numpy as np

from .... models.solow import ces

params = {'A0': 1.0, 'g': 0.02, 'L0': 1.0, 'n': 0.02, 's': 0.15,
          'alpha': 0.33, 'sigma': 1.1, 'delta': 0.05}
model = ces.CESModel(params)


def test_steady_state():
    """Compare analytic steady state with numerical steady state."""
    eps = 1e-1
    for g in np.linspace(eps, 0.05, 4):
        for n in np.linspace(eps, 0.05, 4):
            for s in np.linspace(eps, 1-eps, 4):
                for alpha in np.linspace(eps, 1-eps, 4):
                    for delta in np.linspace(eps, 1-eps, 4):
                        for sigma in np.linspace(eps, 2.0, 4):

                            tmp_params = {'A0': 1.0, 'g': g, 'L0': 1.0, 'n': n,
                                          's': s, 'alpha': alpha, 'delta': delta,
                                          'sigma': sigma}
                            try:
                                model.params = tmp_params

                                # use root finder to compute the steady state
                                actual_ss = model.steady_state
                                expected_ss = model.find_steady_state(1e-12, 1e9)

                                # conduct the test (numerical precision limits!)
                                nose.tools.assert_almost_equals(actual_ss,
                                                                expected_ss,
                                                                places=6)

                            # handles params with non finite steady state
                            except AttributeError:
                                continue
