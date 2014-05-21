
from __future__ import division
import numpy as np
from robustlq import RBLQ
from lqcontrol import LQ
import matplotlib.pyplot as plt

if 0:
    R = 1
    Q = 1
    A = 1
    B = -1
    C = 1
    beta  = 0.95
    theta = 1000.0



######### Monopolist Example #############

# == Model parameters == #
a0      = 5
a1      = 0.5
b       = 1
gamma   = 25
beta    = 0.96
c       = 2

theta_bar = 2.0

# == Useful constants == #
m0 = (a0 - c) / (2 * a1)
m1 = b / (2 * a1)

# == Formulate LQ problem == #
Q = gamma
R = [[a1, -a1, 0],
     [-a1, a1, 0],
     [0,   0,  0]]
A = [[0, 0, m0],
     [0, 1, 0],
     [0, 0, 1]]
B = [[0],
     [1],
     [0]]
C = [[m1],
     [0],
     [0]]

rlq = RBLQ(A, B, C, Q, R, beta, theta_bar)
lq = LQ(Q, R, A, B, beta=beta)

f, k, p = rlq.robust_rule()

print f
print rlq.K_to_F(k)

if 0:
    F_opt, K_opt, P_opt = rlq.robust_rule_simple()
    x0 = np.asarray((1, 0, 1)).reshape(3, 1)

    num_thetas = 20
    thetas = np.linspace(1, 5, num_thetas)
    vals = np.empty((2, num_thetas))
    ent = np.empty(num_thetas)

    for i, theta in enumerate(thetas):
        for theta in (theta, -theta):
            rlq.theta = theta
            F, P, K = rlq.robust_rule()
            vals[i] = rlq.compute_value(F_opt, K, x0)
        ent[i] = rlq.compute_entropy(F, K, x0)

    fig, ax = plt.subplots()
    ax.plot(ent, vals)
    plt.show()
