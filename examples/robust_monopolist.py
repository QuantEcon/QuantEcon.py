"""
Filename: robust_monopolist.py
Authors: Chase Coleman, Spencer Lyon, Thomas Sargent, John Stachurski

The robust control problem for a monopolist with adjustment costs.  The
inverse demand curve is:

  p_t = a_0 - a_1 y_t + d_t

where d_{t+1} = \rho d_t + \sigma_d w_{t+1} for w_t ~ N(0, 1) and iid.
The period return function for the monopolist is

  r_t =  p_t y_t - gamma (y_{t+1} - y_t)^2 / 2 - c y_t

The objective of the firm is E_t \sum_{t=0}^\infty \beta^t r_t

For the linear regulator, we take the state and control to be

    x_t = (1, y_t, d_t) and u_t = y_{t+1} - y_t

"""
import pandas as pd
import numpy as np
from scipy.linalg import eig
from scipy import interp
import matplotlib.pyplot as plt

import quantecon as qe

# == model parameters == #

a_0     = 100
a_1     = 0.5
rho     = 0.9
sigma_d = 0.05
beta    = 0.95
c       = 2
gamma   = 50.0

theta = 0.002
ac    = (a_0 - c) / 2.0

# == Define LQ matrices == #

R = np.array([[0.,  ac,  0.],
              [ac, -a_1, 0.5],
              [0., 0.5,  0.]])

R = -R  # For minimization
Q = gamma / 2

A = np.array([[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., rho]])
B = np.array([[0.],
              [1.],
              [0.]])
C = np.array([[0.],
              [0.],
              [sigma_d]])

# -------------------------------------------------------------------------- #
#                                 Functions
# -------------------------------------------------------------------------- #


def evaluate_policy(theta, F):
    """
    Given theta (scalar, dtype=float) and policy F (array_like), returns the
    value associated with that policy under the worst case path for {w_t}, as
    well as the entropy level.
    """
    rlq = qe.robustlq.RBLQ(Q, R, A, B, C, beta, theta)
    K_F, P_F, d_F, O_F, o_F = rlq.evaluate_F(F)
    x0 = np.array([[1.], [0.], [0.]])
    value = - x0.T.dot(P_F.dot(x0)) - d_F
    entropy = x0.T.dot(O_F.dot(x0)) + o_F
    return list(map(float, (value, entropy)))


def value_and_entropy(emax, F, bw, grid_size=1000):
    """
    Compute the value function and entropy levels for a theta path
    increasing until it reaches the specified target entropy value.

    Parameters
    ==========
    emax: scalar
        The target entropy value

    F: array_like
        The policy function to be evaluated

    bw: str
        A string specifying whether the implied shock path follows best
        or worst assumptions. The only acceptable values are 'best' and
        'worst'.

    Returns
    =======
    df: pd.DataFrame
        A pandas DataFrame containing the value function and entropy
        values up to the emax parameter. The columns are 'value' and
        'entropy'.

    """
    if bw == 'worst':
        thetas = 1 / np.linspace(1e-8, 1000, grid_size)
    else:
        thetas = -1 / np.linspace(1e-8, 1000, grid_size)

    df = pd.DataFrame(index=thetas, columns=('value', 'entropy'))

    for theta in thetas:
        df.ix[theta] = evaluate_policy(theta, F)
        if df.ix[theta, 'entropy'] >= emax:
            break

    df = df.dropna(how='any')
    return df


# -------------------------------------------------------------------------- #
#                                    Main
# -------------------------------------------------------------------------- #


# == Compute the optimal rule == #
optimal_lq = qe.lqcontrol.LQ(Q, R, A, B, C, beta)
Po, Fo, do = optimal_lq.stationary_values()

# == Compute a robust rule given theta == #
baseline_robust = qe.robustlq.RBLQ(Q, R, A, B, C, beta, theta)
Fb, Kb, Pb = baseline_robust.robust_rule()

# == Check the positive definiteness of worst-case covariance matrix to == #
# == ensure that theta exceeds the breakdown point == #
test_matrix = np.identity(Pb.shape[0]) - np.dot(C.T, Pb.dot(C)) / theta
eigenvals, eigenvecs = eig(test_matrix)
assert (eigenvals >= 0).all(), 'theta below breakdown point.'


emax = 1.6e6

optimal_best_case = value_and_entropy(emax, Fo, 'best')
robust_best_case = value_and_entropy(emax, Fb, 'best')
optimal_worst_case = value_and_entropy(emax, Fo, 'worst')
robust_worst_case = value_and_entropy(emax, Fb, 'worst')

fig, ax = plt.subplots()

ax.set_xlim(0, emax)
ax.set_ylabel("Value")
ax.set_xlabel("Entropy")
ax.grid()

for axis in 'x', 'y':
    plt.ticklabel_format(style='sci', axis=axis, scilimits=(0, 0))

plot_args = {'lw': 2, 'alpha': 0.7}

colors = 'r', 'b'

df_pairs = ((optimal_best_case, optimal_worst_case),
            (robust_best_case, robust_worst_case))


class Curve(object):

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __call__(self, z):
        return interp(z, self.x, self.y)


for c, df_pair in zip(colors, df_pairs):
    curves = []
    for df in df_pair:
        # == Plot curves == #
        x, y = df['entropy'], df['value']
        x, y = (np.asarray(a, dtype='float') for a in (x, y))
        egrid = np.linspace(0, emax, 100)
        curve = Curve(x, y)
        print(ax.plot(egrid, curve(egrid), color=c, **plot_args))
        curves.append(curve)
    # == Color fill between curves == #
    ax.fill_between(egrid,
                    curves[0](egrid),
                    curves[1](egrid),
                    color=c, alpha=0.1)

plt.show()
