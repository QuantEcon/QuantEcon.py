"""
Authors: Chase Coleman, Spencer Lyon, Tom Sargent and John Stachurski

The robust control problem for a monopolist with adjustment costs.  The
inverse demand curve is:

  p_t = a_0 - a_1 q_t + d_t

where

  d_{t+1} = \rho d_t + \sigma_d epsilon_{t+1} 
  
for epsilon_t ~ N(0,1) and iid.  The period return function for the monopolist
is

  r_t =  p_t q_t - e (q_{t+1} - q_t) / 2 - c q_t   

The objective of the firm is

  E_t \sum_{t=0}^\infty \beta^t r_t

For the linear regulator, we take the state and control to be

    x_t = (1, q_t, d_t)

    u_t = q_{t+1} - q_t 

"""

from __future__ import division
from robustlq import RBLQ
from lqcontrol import LQ
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import eig


#-----------------------------------------------------------------------------#
# Define model parameters 
#-----------------------------------------------------------------------------#

a_0 = 100
a_1 = 0.5
rho = 0.9
sigma_d = 0.05
beta = 0.95
c = 2
e = 50.0

theta =  1 / 50.0

#-----------------------------------------------------------------------------#
# Define LQ matrices
#-----------------------------------------------------------------------------#

ac = (a_0 - c) / 2.0
R = np.array([[0,  ac,    0], 
              [ac, -a_1, 0.5], 
              [0., 0.5,  0]])
R = -R  # For minimization
Q = e / 2

A = np.array([[1., 0., 0.], 
              [0., 1., 0.], 
              [0., 0., rho]])
B = np.array([[0.], 
              [1.], 
              [0.]])
C = np.array([[0.], 
              [0.], 
              [sigma_d]])

#-----------------------------------------------------------------------------#
# Evaluation
#-----------------------------------------------------------------------------#

# Compute the optimal rule
optimal_lq = LQ(Q, R, A, B, C, beta)
Po, Fo, do = optimal_lq.stationary_values()

# Compute a robust rule given theta
baseline_robust = RBLQ(A, B, C, Q, R, beta, theta)
Fb, Kb, Pb = baseline_robust.robust_rule()

# Check the positive definiteness of the worst-case covariance matrix to
# ensure that theta exceeds the breakdown point
check = np.identity(Pb.shape[0]) - np.dot(C.T, Pb.dot(C)) / theta
checkfinal = eig(check)[0]
if (checkfinal < 0).any():
    raise ValueError("theta below breakdown point. Rechoose parameters.")


def evaluate_policy(theta, F):
    """
    Given theta (scalar, dtype=float), the function returns the optimal value
    function and entropy level.
    """
    rlq = RBLQ(A, B, C, Q, R, beta, theta)
    K_F, P_F, d_F, O_F, o_F = rlq.evaluate_F(F)
    x0 = np.array([[1.], [0.], [0.]])
    value = - x0.T.dot(P_F.dot(x0)) - d_F
    entropy = x0.T.dot(O_F.dot(x0)) + o_F
    return map(float, (value, entropy))


def value_and_entropy(entropy_target, F, bw, grid_size=100):
    """
    Compute the value function and entropy levels for a theta path
    increasing until it reaches the specified target entropy value.

    Parameters
    ==========
    entropy_target : scalar
        The target entropy value

    F : array_like
        The policy function to be evaluated

    bw : str
        A string specifying whether the implied shock path follows best
        or worst assumptions. The only acceptable values are 'best' and
        'worst'.

    Returns
    =======
    df : pd.DataFrame
        A pandas DataFrame containing the value function and entropy
        values up to the entropy_target parameter. The index is the
        relevant portion of sig_vec and the columns are 'value' and
        'entropy'.

    """
    if bw == 'worst':
        thetas = 1 / np.linspace(1e-7, 100, grid_size)
    else:
        thetas = -1 / np.linspace(1e-7, 100, grid_size)

    df = pd.DataFrame(index=thetas, columns=('value', 'entropy'))

    for theta in thetas:
        df.ix[theta] = evaluate_policy(theta, F)
        if df.ix[theta, 'entropy'] >= entropy_target:
            break

    return df


#-----------------------------------------------------------------------------#
#                                    Main
#-----------------------------------------------------------------------------#

entropy_target = 1.6e6

optimal_best_case = value_and_entropy(entropy_target, Fo, 'best')
robust_best_case = value_and_entropy(entropy_target, Fb, 'best')
optimal_worst_case = value_and_entropy(entropy_target, Fo, 'worst')
robust_worst_case = value_and_entropy(entropy_target, Fb, 'worst')

fig, ax = plt.subplots()

#ax.set_xlim(0, entropy_target)
ax.set_ylabel("Value")
ax.set_xlabel("Entropy")

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plot_args = {'color' : 'r',
             'lw' : 2, 
             'alpha' : 0.7} 

for df in (optimal_best_case, optimal_worst_case):
#for df in (optimal_worst_case,):
    x = df['entropy']
    y = df['value']
    print x
    print y
    print ax.plot(x, y, **plot_args)

plot_args['color'] = 'b'
for df in (robust_best_case, robust_worst_case):
#for df in (robust_worst_case,):
    x = df['entropy']
    y = df['value']
    print x
    print y
    print ax.plot(x, y, **plot_args)


plt.show()

