# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 10:26:11 2015

@author: dgevans
"""
import numpy as np
from scipy.optimize import root
from scipy.optimize import fmin_slsqp
import utilities
import lucas_stokey as LS
from utilities import simulate_markov

class Planners_Allocation_Bellman(object):
    '''
    Compute the planner's allocation by solving Bellman
    equation.
    '''
    def __init__(self,Para,mugrid):
        '''
        Initializes the class from the calibration Para
        '''
        self.beta = Para.beta
        self.Pi = Para.Pi
        self.G = Para.G
        self.S = len(Para.Pi) # number of states
        self.Theta = Para.Theta
        self.Para = Para
        self.mugrid = mugrid
        
        #now find the first best allocation
        self.solve_time1_bellman()
        self.T.time_0 = True #Bellman equation now solves time 0 problem
        
    def solve_time1_bellman(self):
        '''
        Solve the time  1 Bellman equation for calibration Para and initial grid mugrid0
        '''
        Para,mugrid0 = self.Para,self.mugrid
        Pi = Para.Pi
        S = len(Para.Pi)
        
        #First get initial fit from lucas stockey solution.  
        #Need to change things to be ex_ante
        PP = LS.Planners_Allocation_Sequential(Para)
        interp = utilities.interpolator_factory(2,None)
        
        def incomplete_allocation(mu_,s_):
            c,n,x,V = PP.time1_value(mu_)
            return c,n,Pi[s_].dot(x),Pi[s_].dot(V)
        cf,nf,xgrid,Vf,xprimef = [],[],[],[],[]
        for s_ in range(S):
            c,n,x,V = zip(*map(lambda mu: incomplete_allocation(mu,s_),mugrid0)) 
            c,n = np.vstack(c).T,np.vstack(n).T
            x,V = np.hstack(x),np.hstack(V)
            xprimes = np.vstack([x]*S)
            cf.append(interp(x,c))
            nf.append(interp(x,n))
            Vf.append(interp(x,V))
            xgrid.append(x)
            xprimef.append(interp(x,xprimes))
        cf,nf,xprimef = utilities.fun_vstack(cf), utilities.fun_vstack(nf),utilities.fun_vstack(xprimef)
        Vf = utilities.fun_hstack(Vf)
        policies = [cf,nf,xprimef]            
        
        
        #create xgrid    
        x = np.vstack(xgrid).T
        xbar = [x.min(0).max(),x.max(0).min()]
        xgrid = np.linspace(xbar[0],xbar[1],len(mugrid0))
        self.xgrid = xgrid
                
        #Now iterate on Bellman equation
        T = BellmanEquation(Para,xgrid,policies)
        diff = 1.
        while diff > 1e-6:
            PF = T(Vf)
        
            Vfnew,policies = self.fit_policy_function(PF)
            diff = np.abs((Vf(xgrid)-Vfnew(xgrid))/Vf(xgrid)).max()
            
            print diff
            Vf = Vfnew
            
        #store value function policies and Bellman Equations
        self.Vf = Vf
        self.policies = policies
        self.T = T
        
    def fit_policy_function(self,PF):
        '''
        Fits the policy functions
        '''
        S,xgrid = len(self.Pi),self.xgrid
        interp = utilities.interpolator_factory(3,0)
        cf,nf,xprimef,Tf,Vf = [],[],[],[],[]
        for s_ in range(S):
            PFvec = np.vstack([PF(x,s_) for x in self.xgrid]).T
            Vf.append(interp(xgrid,PFvec[0,:]))
            cf.append(interp(xgrid,PFvec[1:1+S]))
            nf.append(interp(xgrid,PFvec[1+S:1+2*S]))
            xprimef.append(interp(xgrid,PFvec[1+2*S:1+3*S]))
            Tf.append(interp(xgrid,PFvec[1+3*S:]))
        policies = utilities.fun_vstack(cf), utilities.fun_vstack(nf),utilities.fun_vstack(xprimef),utilities.fun_vstack(Tf)
        Vf = utilities.fun_hstack(Vf)
        return Vf,policies
        
    def Tau(self,c,n):
        '''
        Computes Tau given c,n
        '''
        Para = self.Para
        Uc,Un = Para.Uc(c,n),Para.Un(c,n)
        
        return 1+Un/(self.Theta * Uc)
        
    def time0_allocation(self,B_,s0):
        '''
        Finds the optimal allocation given initial government debt B_ and state s_0
        '''
        PF = self.T(self.Vf)
        
        z0 = PF(B_,s0)
        c0,n0,xprime0,T0 = z0[1:]
        return c0,n0,xprime0,T0
        
    def simulate(self,B_,s_0,T,sHist=None):
        '''
        Simulates planners policies for T periods
        '''
        Para,Pi = self.Para,self.Pi
        Uc = Para.Uc
        cf,nf,xprimef,Tf = self.policies
        
        if sHist == None:
            sHist = simulate_markov(Pi,s_0,T)
        
        cHist,nHist,Bhist,xHist,TauHist,THist,muHist = np.zeros((7,T))
        #time0
        cHist[0],nHist[0],xHist[0],THist[0]  = self.time0_allocation(B_,s_0)
        TauHist[0] = self.Tau(cHist[0],nHist[0])[s_0]
        Bhist[0] = B_
        muHist[0] = self.Vf[s_0](xHist[0])
        
        #time 1 onward
        for t in range(1,T):
            s_,x,s = sHist[t-1],xHist[t-1],sHist[t]
            c,n,xprime,T = cf[s_,:](x),nf[s_,:](x),xprimef[s_,:](x),Tf[s_,:](x)
            
            Tau = self.Tau(c,n)[s]
            u_c = Uc(c,n)
            Eu_c = Pi[s_,:].dot(u_c)
            
            muHist[t] = self.Vf[s](xprime[s])
            
            cHist[t],nHist[t],Bhist[t],TauHist[t] = c[s],n[s],x/Eu_c,Tau
            xHist[t],THist[t] = xprime[s],T[s]
        return cHist,nHist,Bhist,xHist,TauHist,THist,muHist,sHist
        
        
class BellmanEquation(object):
    '''
    Bellman equation for the continuation of the Lucas-Stokey Problem
    '''
    def __init__(self,Para,xgrid,policies0):
        '''
        Initializes the class from the calibration Para
        '''
        self.beta = Para.beta
        self.Pi = Para.Pi
        self.G = Para.G
        self.S = len(Para.Pi) # number of states
        self.Theta = Para.Theta
        self.Para = Para
    
        self.xbar = [min(xgrid),max(xgrid)]
        self.time_0 = False
        
        self.z0 = {}
        cf,nf,xprimef = policies0
        
        for s_ in range(self.S):
            for x in xgrid:
                self.z0[x,s_] = np.hstack([cf[s_,:](x),nf[s_,:](x),xprimef[s_,:](x),np.zeros(self.S)])
                
        self.find_first_best()
                
    def find_first_best(self):
        '''
        Find the first best allocation
        '''
        Para = self.Para
        S,Theta,Uc,Un,G = self.S,self.Theta,Para.Uc,Para.Un,self.G
        
        def res(z):
            c = z[:S]
            n = z[S:]
            return np.hstack(
            [Theta*Uc(c,n)+Un(c,n), Theta*n - c - G]            
            )
        res = root(res,0.5*np.ones(2*S))
        if not res.success:
            raise Exception('Could not find first best')
        
        self.cFB = res.x[:S]
        self.nFB = res.x[S:]
        IFB = Uc(self.cFB,self.nFB)*self.cFB + Un(self.cFB,self.nFB)*self.nFB
        
        self.xFB = np.linalg.solve(np.eye(S) - self.beta*self.Pi, IFB)
        
        self.zFB = {}
        for s in range(S):
            self.zFB[s] = np.hstack([self.cFB[s],self.nFB[s],self.Pi[s].dot(self.xFB),0.])
                
            
        
    def __call__(self,Vf):
        '''
        Given continuation value function next period return value function this
        period return T(V) and optimal policies
        '''
        if not self.time_0:
            PF = lambda x,s: self.get_policies_time1(x,s,Vf)
        else:
            PF = lambda B_,s0: self.get_policies_time0(B_,s0,Vf)
        return PF
    
    def get_policies_time1(self,x,s_,Vf):
        '''
        Finds the optimal policies 
        '''
        Para,beta,Theta,G,S,Pi = self.Para,self.beta,self.Theta,self.G,self.S,self.Pi
        U,Uc,Un = Para.U,Para.Uc,Para.Un
        
        def objf(z):
            c,n,xprime = z[:S],z[S:2*S],z[2*S:3*S]
            
            Vprime = np.empty(S)
            for s in range(S):
                Vprime[s] = Vf[s](xprime[s])

            return -Pi[s_].dot(U(c,n)+beta*Vprime)
            
        def cons(z):
            c,n,xprime,T = z[:S],z[S:2*S],z[2*S:3*S],z[3*S:]
            u_c = Uc(c,n)
            Eu_c = Pi[s_].dot(u_c)
            return np.hstack([
            x*u_c/Eu_c - u_c*(c-T)-Un(c,n)*n - beta*xprime,
            Theta*n - c - G          
            ])
            
        if Para.transfers:
            bounds = [(0.,100)]*S+[(0.,100)]*S+[self.xbar]*S+[(0.,100.)]*S
        else:
            bounds = [(0.,100)]*S+[(0.,100)]*S+[self.xbar]*S+[(0.,0.)]*S
        out,fx,_,imode,smode = fmin_slsqp(objf,self.z0[x,s_],f_eqcons=cons,
                            bounds=bounds,full_output=True,iprint=0)
                            
        if imode >0:
            raise Exception(smode)
           
        self.z0[x,s_] = out
        return np.hstack([-fx,out])
        
    def get_policies_time0(self,B_,s0,Vf):
        '''
        Finds the optimal policies 
        '''
        Para,beta,Theta,G = self.Para,self.beta,self.Theta,self.G
        U,Uc,Un = Para.U,Para.Uc,Para.Un
        
        def objf(z):
            c,n,xprime = z[:-1]

            return -(U(c,n)+beta*Vf[s0](xprime))
            
        def cons(z):
            c,n,xprime,T = z
            return np.hstack([
            -Uc(c,n)*(c-B_-T)-Un(c,n)*n - beta*xprime,
            (Theta*n - c - G)[s0]            
            ])
            
        if Para.transfers:
            bounds=[(0.,100),(0.,100),self.xbar,(0.,100.)]
        else:
            bounds=[(0.,100),(0.,100),self.xbar,(0.,0.)]
        out,fx,_,imode,smode = fmin_slsqp(objf,self.zFB[s0],f_eqcons=cons,
                            bounds=bounds,full_output=True,iprint=0)
                            
        if imode >0:
            raise Exception(smode)
           
        return np.hstack([-fx,out])
