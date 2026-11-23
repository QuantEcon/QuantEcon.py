"""Check fixed and variable IVP trajectories around row-batch boundaries."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from quantecon import IVP


@pytest.mark.parametrize('steps', [1, 254, 255, 256, 511, 512])
@pytest.mark.parametrize('variable', [False, True])
def test_trajectory_at_batch_boundaries(steps, variable):
    dt = 2 ** -12
    y0 = np.array([1.0, -0.25])

    def rhs(t, y):
        return np.array([2 * t, -1])

    if variable:
        def event(t, y):
            return (steps - 0.5) * dt - t
        stop = dict(g=event, tol=0)
    else:
        stop = dict(T=steps * dt)

    result = IVP(rhs).solve(0.0, y0, h=dt, **stop)
    time = np.arange(steps + 1) * dt
    assert result.shape == (steps + 1, 3)
    assert_array_equal(result[:, 0], time)
    assert_allclose(result[:, 1:], np.c_[y0[0] + time**2, y0[1] - time],
                    rtol=0, atol=2e-12)


@pytest.mark.parametrize('variable', [False, True])
def test_trajectory_with_no_accepted_steps_preserves_shape(variable):
    class UnableToStart(IVP):
        def successful(self):
            return False

    model = UnableToStart(lambda t, y: -y)
    y0 = np.array([1.0, 2.0])
    stop = dict(g=lambda t, y: 0, tol=0) if variable else dict(T=1)
    result = model.solve(0, y0, **stop)
    assert result.shape == (3,)
    assert_array_equal(result, [0.0, 1.0, 2.0])
