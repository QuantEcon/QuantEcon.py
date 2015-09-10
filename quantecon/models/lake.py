# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:08:44 2015

Author: David Evans

Provides a class call LakeModel that simulates the dynamics of unemployment and 
employment. 
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import quantecon as qe

class LakeModel(object):
    r"""
    This class solves the lake model and simulates :math:`E_t` and :math:`U_t`
    with the following parameters
    
    Parameters:
    ------------
    lamb: The job finding rate for currently unemployed workers.

    alpha: The dismissal rate for currently employed workers.

    b : Entry rate into the labor force.

    d : Exit rate from the labor force.
    
    Attributes
    ------------
    
    In solving the lake model the program computes
    
    A : Matrix governing law of motion for :math:`E_t` and :math:`U_t`
    
    hatA : Matrix governing the law of motion for the rates :math:`e_t`
    and :math:`u_t`
    
    """
    def __init__(self,lamb,alpha,b,d):
        self.lamb = lamb
        self.alpha= alpha
        self.b = b
        self.d = d
        self.g = b-d
        
        self.A = self.ConstructA_matrix()
        self.Ahat = self.Construct_ScaledA_matrix()
        
    def ConstructA_matrix(self):
        r'''
        Constructs the A matrix for :math:`X_{t+1} = A X_{t}` where :math:`X_t = (E_t,U_t)`
    
        Returns
        --------
            A : (2x2) matrix governing state dynamics
        '''
        lamb,alpha,b,d = self.lamb,self.alpha,self.b,self.d
        
        return np.array([ [(1-d)*(1-alpha), (1-d)*lamb],
                          [(1-d)*alpha + b, (1-lamb)*(1-d) + b ]])
    
    def Construct_ScaledA_matrix(self):
        r'''
        Constructs the scaled A matrix for :math:`x_{t+1} = Ahat x_{t}` where 
        :math:`x_t = (E_t/N_t,U_t/N_t) = (e_t,u_t)`
            
        Returns
        --------
            Ahat : (2x2) matrix governing state dynamics for employment rates.
        '''
        A = self.ConstructA_matrix()
        return A/(1+self.g)
        
    def find_steady_state(self):
        r"""
        Finds the steady state of the system :math:`x_{t+1} = \hat A x_{t}`
        
        Returns
        --------
        
            xbar : steady state vector of employment and unemployment rates
        """
        x = np.ones(2)/2.
        diff = 1.
        while diff > 1e-6:
            xprime = self.Ahat.dot(x)
            diff = np.abs(xprime-x).max()
            x = xprime
            
        return x
        
    def simulate_stock_path(self,X0,T):
        r'''
        Simulates the the sequence of Employment and Unemployent stocks
        
        Parameters
        ------------
        
        X0 : array containing initial values (E0,U0)
        
        T : Number of periods to simulate
        
        Returns
        ---------
        
        X : iterator containing sequence of Employment and Unemployment stocks
        '''
        X = np.atleast_1d(X0) # recast as array just in case
        for t in range(T):
            yield X
            X = self.A.dot(X)
            
    def simulate_rate_path(self,x0,T):
        r'''
        Simulates the the sequence of Employment and Unemployent rates
        
        Parameters
        ------------
        
        x0 : array containing initial values (e0,u0)
        
        T : Number of periods to simulate
        
        Returns
        ---------
        
        x : iterator containing sequence of Employment and Unemployment rates
        '''
        x = np.atleast_1d(x0) # recast as array just in case
        for t in range(T):
            yield x
            x = self.Ahat.dot(x)
            

            
class LakeModelAgent(object):
    r'''
    This class holds methods necessary to simulate the life course of an agent
    who lives in the lake model economy with the following parameters
    
    Parameters:
    ------------
    lamb: The job finding rate if agent is currently unemployed.

    alpha: The dismissal rate if agent is currently employed.
    '''
    def __init__(self,lamb,alpha):
        self.lamb = lamb
        self.alpha = alpha
        
        self.P = self.compute_P()
        
    def compute_P(self):
        r'''
        This method computes the transition matrix for the agent.
        
        Return
        ---------
        
        P(2d-array) : Transition matrix for the agent
        '''
        alpha,lamb = self.alpha,self.lamb
        return np.array([
        [(1-alpha), alpha],
        [lamb, 1-lamb]        
        ])
        
    def compute_ergodic(self):
        r'''
        Computes the ergodic distribution over the unemployment and employment
        states
        
        Returns
        ---------
        
        pibar(1d-array) : the ergodic distribution of P
        '''
        return qe.mc_compute_stationary(self.P)
        
    def simulate(self,s0,T):
        r'''
        Simulates the life of an agent for T periods
        
        Parameters
        -------------
        
        s0(int) : initial state
        
        T(int) : number of periods to simulate
        
        Returns
        -----------
        
        sHist(iterator) : history of employment(s==0) and unemployment(s==1)
        '''
        pi0 = np.arange(2) == s0
        return qe.mc_sample_path(self.P,pi0,T)
            
            
class LakeModel_Equilibrium(object):
    r'''
    Solves for the steady state General Equilibirium of a Lake Model economy
    using the McCall Search Model to model the behavior 
    
    Parameters
    -------------
    
    alpha - (float) Exogenous firing rate.
    
    beta - (float) The discount factor.
    
    gamma - (float) Arrival rate of wage offer
    
    sigma - (float) Degree of risk aversion.
    
    pdf - (1d - array) pdf[s] Probability of receiving a wage wstar[s]
    
    wstar - (1d -array) Distribution of wages.
    '''        
    
    def __init__(self,alpha,gamma,beta,sigma,pdf,wstar):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.pdf = pdf
        self.wstar = wstar
        
    def U(self,c):
        r'''
        Utility function of the agent
        
        Parameters
        -------------
        
        c - (array or float) consumption of the agent
        
        Returns
        ----------
        
        U - (array or float) Utility of the agent for each level of consumption
        '''
        sigma = self.sigma
        negative_c = c < 0.
        if sigma == 1.:
            U =  np.log(c)
        else:
            U= (c**(1-sigma) - 1)/(1-sigma)
        U[negative_c] = -9999999.
        return U
        
            
            
    def solve_for_steadystate(self,c,T):
        r'''
        Solve for workers steady state policies given tax policy
        
        Paramaters
        ------------
        
        c - (float) Level of unemployment benefit
        
        T - (float) Lump sum tax
        
        Results
        ---------

        V - (array) Value function of the worker        
        
        C - (array) optimal policy function of the worker
        
        pi - (array) distribution between employed and unemployed
        
        W - (float) Welfare at steady state
        '''
        
        V,C = self.solveMcCallModel(c-T,self.wstar-T)
        
        U = self.U(np.array([c-T])) + self.beta*self.pdf.dot(V) # value of unemployment
        
        lamb = self.gamma * self.pdf.dot(C) # probability of accepting a job
        
        LM = LakeModel(lamb,self.alpha,0,0) #no birth or death
        
        pi = LM.find_steady_state()
        
        #Expected value of being employed
        EV = (C*V).dot(self.pdf)/(C.dot(self.pdf)) 
        
        W = pi[0]*EV + pi[1] * U
        
        return V,C,pi,W,U,EV
        
    def find_steady_state_tax(self,c):
        r'''
        Finds the lump sum tax that balances budget at steady state
        
        Parameters
        -----------
        
        c - (float) Level of unemployment benefit
        
        Results
        ----------
        
        T - (float) Lump sum tax that balances budget
        
        W - (float) Steady State Wealfare at that balanced budget tax
        '''
        
        #budget at steady state
        def SS_budget(T):
            V,C,pi,W,U,EV= self.solve_for_steadystate(c,T)
            #return inflows minus outflows
            return T - pi[1]*c
            
        T = brentq(SS_budget, 0., 0.9*c)
        
        V,C,pi,W,U,EV = self.solve_for_steadystate(c,T)
        
        return T,W,U,EV,pi

        
    def iterateValueFunction(self,c,w,V): 
        r'''
        Iterates McCall search value function v
        
        Parameters
        ----------

        c - (float) Level of unemployment insurance

        w - (float) Vector of possible wages        
        
        V - (n array) continuation value function if offered w[s] next period
        
        Returns
        --------

        V_new - (n array) current value function if offered w[s] this period 

        Choice - (n array) do we accept or reject wage w[s]
        '''
        p,beta,alpha,gamma = self.pdf,self.beta,self.alpha,self.gamma
        S = len(p)
        Q = p.dot(V)# value before wage offer
        V_U = (self.U(c*np.ones(S)) + beta*gamma*Q)/( 1-beta*(1-gamma) )
        #stack value of accepting and rejecting offer on top of each other
        stacked_values = np.vstack([ V_U,
                                    self.U(w) + (1-alpha)*beta*V + alpha*beta*V_U ]) 
                                    
        #find whether it is optimal to accept or reject offer    
        V_new = np.amax(stacked_values, axis = 0) 
        Choice = np.argmax(stacked_values, axis = 0)
        return V_new,Choice
        
    def solveMcCallModel(self,c,w,eps = 1e-6): 
        r'''
        Solves the infinite horizon McCall search model
        
        Parameters
        -----------
        
        c - (float) Level of unemployment insurance

        w - (float) Vector of possible wages 
        
        eps - (float) convergence criterion for infinite horizon
        
        Returns
        --------
        
        V - (1d-array) Value function that solves the infinite horizon problem
        
        Choice - (1d-array) Optimal policy of workers
        '''
        
        S = len(self.pdf)
        v = np.zeros(S) #intialize with zero
        diff = 1 #holds difference v_{t+1}-v_t
        while diff > eps:
            v_new,choice = self.iterateValueFunction(c,w,v) 
            diff = np.amax( np.abs(v-v_new) )#compute difference between value
            v = v_new #copy v_new into v #add in infinte horizon solution 
            
        return v,choice
    
    
    

    