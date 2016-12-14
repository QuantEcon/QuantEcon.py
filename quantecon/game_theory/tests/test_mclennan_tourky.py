"""
Author: Daisuke Oyama

Tests for mclennan_tourky.py

"""
import numpy as np
from nose.tools import ok_
from quantecon.game_theory import Player, NormalFormGame, mclennan_tourky


class TestMclennanTourky():
    def setUp(self):
        def anti_coordination(N, v):
            payoff_array = np.empty((2,)*N)
            payoff_array[0, :] = 1
            payoff_array[1, :] = 0
            payoff_array[1].flat[0] = v
            g = NormalFormGame((Player(payoff_array),)*N)
            return g

        def p_star(N, v):
            # Unique symmetric NE mixed action: [p_star, 1-p_star]
            return 1 / (v**(1/(N-1)))

        def epsilon_nash_interval(N, v, epsilon):
            # Necessary, but not sufficient, condition: lb < p < ub
            lb = p_star(N, v) - epsilon / ((N-1)*(v**(1/(N-1))-1))
            ub = p_star(N, v) + epsilon / (N-1)
            return lb, ub

        self.game_dicts = []
        v = 2
        epsilon = 1e-5

        Ns = [2, 3, 4]
        for N in Ns:
            g = anti_coordination(N, v)
            lb, ub = epsilon_nash_interval(N, v, epsilon)
            d = {'g': g,
                 'epsilon': epsilon,
                 'lb': lb,
                 'ub': ub}
            self.game_dicts.append(d)

    def test_convergence_default(self):
        for d in self.game_dicts:
            NE, res = mclennan_tourky(d['g'], full_output=True)
            ok_(res.converged)

    def test_pure_nash(self):
        for d in self.game_dicts:
            init = (1,) + (0,)*(d['g'].N-1)
            NE, res = mclennan_tourky(d['g'], init=init, full_output=True)
            ok_(res.num_iter==1)

    def test_epsilon_nash(self):
        for d in self.game_dicts:
            NE, res = \
                mclennan_tourky(d['g'], epsilon=d['epsilon'], full_output=True)
            for i in range(d['g'].N):
                ok_(d['lb'] < NE[i][0] < d['ub'])


if __name__ == '__main__':
    import sys
    import nose

    argv = sys.argv[:]
    argv.append('--verbose')
    argv.append('--nocapture')
    nose.main(argv=argv, defaultTest=__file__)
