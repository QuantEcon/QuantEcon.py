from __future__ import division
import numpy as np

class UncertaintyTrapEcon(object):

    def __init__(self,
                a=1.5,          # Risk aversion
                gx=0.5,         # Production shock precision
                rho=0.99,       # Correlation coefficient for theta
                sig_theta=0.5,  # Std dev of theta shock
                num_firms=100,  # Number of firms
                sig_F=1.5,      # Std dev of fixed costs
                c=-420,         # External opportunity cost
                mu_init=0,      # Initial value for mu
                gamma_init=4,   # Initial value for gamma
                theta_init=0):  # Initial value for theta

        # == Record values == #
        self.a, self.gx, self.rho, self.sig_theta = a, gx, rho, sig_theta
        self.num_firms, self.sig_F, self.c, = num_firms, sig_F, c
        self.sd_x = np.sqrt(1/ gx)

        # == Initialize states == #
        self.gamma, self.mu, self.theta =  gamma_init, mu_init, theta_init

    def psi(self, F):
        temp1 = -self.a * (self.mu - F) 
        temp2 = self.a**2 * (1/self.gamma + 1/self.gx) / 2
        return (1 / self.a) * (1 - np.exp(temp1 + temp2)) - self.c

    def update_beliefs(self, X, M):
        """
        Update beliefs (mu, gamma) based on aggregates X and M.
        """
        # Simplify names
        gx, rho, sig_theta = self.gx, self.rho, self.sig_theta
        # Update mu
        temp1 = rho * (self.gamma * self.mu + M * gx * X)
        temp2 = self.gamma + M * gx
        self.mu =  temp1 / temp2
        # Update gamma
        self.gamma = 1 / (rho**2 / (self.gamma + M * gx) + sig_theta**2)

    def update_theta(self, w):
        """
        Update the fundamental state theta given shock w.
        """
        self.theta = self.rho * self.theta + self.sig_theta * w

    def gen_aggregates(self):
        """
        Generate aggregates based on current beliefs (mu, gamma).  This 
        is a simulation step that depends on the draws for F.
        """
        F_vals = self.sig_F * np.random.randn(self.num_firms)
        M = np.sum(self.psi(F_vals) > 0)  # Counts number of active firms
        if M > 0:
            x_vals = self.theta + self.sd_x * np.random.randn(M)
            X = x_vals.mean()
        else:
            X = 0
        return X, M
