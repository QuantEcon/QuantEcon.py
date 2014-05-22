"""
Filename: asset_pricing.py
Authors: David Evans, John Stachurski and Thomas J. Sargent

Computes asset prices in an endowment economy when the endowment obeys
geometric growth driven by a finite state Markov chain.  The transition matrix
of the Markov chain is P, and the set of states is s.  The discount
factor is beta, and gamma is the coefficient of relative risk aversion in the
household's utility function.
"""

import numpy as np
from numpy.linalg import solve

class AssetPrices:

    def __init__(self, beta, P, s, gamma):
        '''
        Initializes an instance of AssetPrices

        Parameters
        ==========
        beta : float
            discount factor 

        P : array_like
            transition matrix 
            
        s : array_like
            growth rate of consumption 

        gamma : float
            coefficient of risk aversion 
        '''
        self.beta, self.gamma = beta, gamma
        self.P, self.s = [np.atleast_2d(x) for x in P, s]
        self.n = self.P.shape[0]
        self.s.shape = self.n, 1

    def tree_price(self):
        '''
        Computes the function v such that the price of the lucas tree is
        v(lambda)C_t
        '''
        # == Simplify names == #
        P, s, gamma, beta = self.P, self.s, self.gamma, self.beta
        # == Compute v == #
        P_tilde = P * s**(1-gamma) #using broadcasting
        I = np.identity(self.n)
        O = np.ones(self.n)
        v = beta * solve(I - beta * P_tilde, P_tilde.dot(O))
        return v
        
    def consol_price(self, zeta):
        '''
        Computes price of a consol bond with payoff zeta

        Parameters
        ===========
        zeta : float
            coupon of the console

        '''
        # == Simplify names == #
        P, s, gamma, beta = self.P, self.s, self.gamma, self.beta
        # == Compute price == #
        P_check = P * s**(-gamma)
        I = np.identity(self.n)
        O = np.ones(self.n)
        p_bar = beta * solve(I - beta * P_check, P_check.dot(zeta * O))
        return p_bar
        
    def call_option(self, zeta, p_s, T=[], epsilon=1e-8):
        '''
        Computes price of a call option on a consol bond with payoff zeta

        Parameters
        ===========
        zeta : float
            coupon of the console

        p_s : float
            strike price 

        T : list of integers 
            length of option 

        epsilon : float
            tolerance for infinite horizon problem
        '''
        # == Simplify names, initialize variables == #
        P, s, gamma, beta = self.P, self.s, self.gamma, self.beta
        P_check = P * s**(-gamma)
        # == Compute consol price == #
        v_bar = self.consol_price(zeta)
        # == Compute option price == #
        w_bar = np.zeros(self.n)
        error = epsilon + 1
        t = 0
        w_bars = {}
        while error > epsilon:
            if t in T:
                w_bars[t] = w_bar
            # == Maximize across columns == #
            to_stack = (beta*P_check.dot(w_bar), v_bar-p_s)
            w_bar_new = np.amax(np.vstack(to_stack), axis = 0 ) 
            # == Find maximal difference of each component == #
            error = np.amax(np.abs(w_bar-w_bar_new)) 
            # == Update == #
            w_bar = w_bar_new
            t += 1
        
        return w_bar, w_bars
