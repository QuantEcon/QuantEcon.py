# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:43:37 2015

@author: dgevans
"""
import numpy as np
from scipy.optimize import root
from scipy.optimize import fmin_slsqp
from scipy.interpolate import UnivariateSpline
from quantecon import compute_fixed_point
from quantecon.markov import mc_sample_path


class Planners_Allocation_Sequential(object):
    '''
    Class returns planner's allocation as a function of the multiplier on the 
    implementability constraint mu
    '''
    def __init__(self,Para):
        '''
        Initializes the class from the calibration Para
        '''
        self.beta = Para.beta
        self.Pi = Para.Pi
        self.G = Para.G
        self.S = len(Para.Pi) # number of states
        self.Theta = Para.Theta
        self.Para = Para
        #now find the first best allocation
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
        self.XiFB = Uc(self.cFB,self.nFB) #multiplier on the resource constraint.
        self.zFB = np.hstack([self.cFB,self.nFB,self.XiFB])
        
    def time1_allocation(self,mu):
        '''
        Computes optimal allocation for time t\geq 1 for a given \mu
        '''
        Para = self.Para
        S,Theta,G,Uc,Ucc,Un,Unn = self.S,self.Theta,self.G,Para.Uc,Para.Ucc,Para.Un,Para.Unn
        def FOC(z):
            c = z[:S]
            n = z[S:2*S]
            Xi = z[2*S:]
            return np.hstack([            
            Uc(c,n) - mu*(Ucc(c,n)*c+Uc(c,n)) -Xi, #foc c
            Un(c,n) - mu*(Unn(c,n)*n+Un(c,n)) + Theta*Xi, #foc n
            Theta*n - c - G #resource constraint
            ])
        
        #find the root of the FOC
        res = root(FOC,self.zFB)
        if not res.success:
            raise Exception('Could not find LS allocation.')
        z = res.x
        c,n,Xi = z[:S],z[S:2*S],z[2*S:]
        
        #now compute x
        I  = Uc(c,n)*c +  Un(c,n)*n
        x = np.linalg.solve(np.eye(S) - self.beta*self.Pi, I )
        
        return c,n,x,Xi        
            
    def time0_allocation(self,B_,s_0):
        '''
        Finds the optimal allocation given initial government debt B_ and state s_0
        '''
        Para,Pi,Theta,G,beta = self.Para,self.Pi,self.Theta,self.G,self.beta
        Uc,Ucc,Un,Unn = Para.Uc,Para.Ucc,Para.Un,Para.Unn
        
        #first order conditions of planner's problem
        def FOC(z):
            mu,c,n,Xi = z
            xprime = self.time1_allocation(mu)[2]
            return np.hstack([
            Uc(c,n)*(c-B_) + Un(c,n)*n + beta*Pi[s_0].dot(xprime),
            Uc(c,n) - mu*(Ucc(c,n)*(c-B_) + Uc(c,n)) - Xi,
            Un(c,n) - mu*(Unn(c,n)*n+Un(c,n)) + Theta[s_0]*Xi,   
            (Theta*n - c - G)[s_0]
            ])
        
        #find root
        res = root(FOC,np.array([0.,self.cFB[s_0],self.nFB[s_0],self.XiFB[s_0]]))
        if not res.success:
            raise Exception('Could not find time 0 LS allocation.')

        return res.x
        
    def time1_value(self,mu):
        '''
        Find the value associated with multiplier mu
        '''
        c,n,x,Xi = self.time1_allocation(mu)
        U = self.Para.U(c,n)
        V = np.linalg.solve(np.eye(self.S) - self.beta*self.Pi, U )
        return c,n,x,V
        
    def Tau(self,c,n):
        '''
        Computes Tau given c,n
        '''
        Para = self.Para
        Uc,Un = Para.Uc(c,n),Para.Un(c,n)
        
        return 1+Un/(self.Theta * Uc)
        
    def simulate(self,B_,s_0,T,sHist=None):
        '''
        Simulates planners policies for T periods
        '''
        Para,Pi,beta = self.Para,self.Pi,self.beta
        Uc = Para.Uc
        
        
        if sHist == None:
            sHist = mc_sample_path(Pi,s_0,T)
        
        cHist,nHist,Bhist,TauHist,muHist = np.zeros((5,T))
        RHist = np.zeros(T-1)
        #time0
        mu,cHist[0],nHist[0],_  = self.time0_allocation(B_,s_0)
        TauHist[0] = self.Tau(cHist[0],nHist[0])[s_0]
        Bhist[0] = B_
        muHist[0] = mu
        
        #time 1 onward
        for t in range(1,T):
            c,n,x,Xi = self.time1_allocation(mu)
            Tau = self.Tau(c,n)
            u_c = Uc(c,n)
            s = sHist[t]
            Eu_c = Pi[sHist[t-1]].dot(u_c)
            
            cHist[t],nHist[t],Bhist[t],TauHist[t] = c[s],n[s],x[s]/u_c[s],Tau[s]
            
            RHist[t-1] = Uc(cHist[t-1],nHist[t-1])/(beta*Eu_c)
            muHist[t] = mu
        
        return cHist,nHist,Bhist,TauHist,sHist,muHist,RHist
            
            
    
  
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
        Solve the time 1 Bellman equation for calibration Para and initial grid mugrid0
        '''
        Para,mugrid0 = self.Para,self.mugrid
        S = len(Para.Pi)
        
        #First get initial fit
        PP = Planners_Allocation_Sequential(Para)
        c,n,x,V = map(np.vstack, zip(*map(lambda mu: PP.time1_value(mu),mugrid0)) )
        
        Vf,cf,nf,xprimef = {},{},{},{}
        for s in range(2):
            cf[s] = UnivariateSpline(x[:,s],c[:,s])
            nf[s] = UnivariateSpline(x[:,s],n[:,s])
            Vf[s] = UnivariateSpline(x[:,s],V[:,s])
            for sprime in range(S):
                xprimef[s,sprime] = UnivariateSpline(x[:,s],x[:,s])
        policies = [cf,nf,xprimef]            
        
        
        #create xgrid    
        xbar = [x.min(0).max(),x.max(0).min()]
        xgrid = np.linspace(xbar[0],xbar[1],len(mugrid0))
        self.xgrid = xgrid
                
        #Now iterate on bellman equation
        T = BellmanEquation(Para,xgrid,policies)
        diff = 1.
        while diff > 1e-5:
            PF = T(Vf)
        
            Vfnew,policies = self.fit_policy_function(PF)
            
            diff = 0.
            for s in range(S):
                diff = max(diff, np.abs((Vf[s](xgrid)-Vfnew[s](xgrid))/Vf[s](xgrid)).max() )
            
            print diff
            Vf = Vfnew
            
        #store value function policies and Bellman Equations
        self.Vf = Vf
        self.policies = policies
        self.T = T
        
    def fit_policy_function(self,PF):
        '''
        Fits the policy functions PF using the points xgrid using UnivariateSpline
        '''
        xgrid,S = self.xgrid,self.S
        
        Vf,cf,nf,xprimef = {},{},{},{}
        for s in range(S):
            PFvec = np.vstack(map(lambda x:PF(x,s),xgrid))
            Vf[s] = UnivariateSpline(xgrid,PFvec[:,0],s=0)
            cf[s] = UnivariateSpline(xgrid,PFvec[:,1],s=0,k=1)
            nf[s] = UnivariateSpline(xgrid,PFvec[:,2],s=0,k=1)
            for sprime in range(S):
                xprimef[s,sprime] = UnivariateSpline(xgrid,PFvec[:,3+sprime],s=0,k=1)
        
        return Vf,[cf,nf,xprimef]
        
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
        c0,n0,xprime0 = z0[1],z0[2],z0[3:]
        return c0,n0,xprime0
        
    def simulate(self,B_,s_0,T,sHist=None):
        '''
        Simulates Ramsey plan for T periods
        '''
        Para,Pi = self.Para,self.Pi
        Uc = Para.Uc
        cf,nf,xprimef = self.policies
        
        if sHist == None:
            sHist = mc_sample_path(Pi,s_0,T)
        
        cHist,nHist,Bhist,TauHist,muHist = np.zeros((5,T))
        RHist = np.zeros(T-1)
        #time0
        cHist[0],nHist[0],xprime  = self.time0_allocation(B_,s_0)
        TauHist[0] = self.Tau(cHist[0],nHist[0])[s_0]
        Bhist[0] = B_
        muHist[0] = 0.
        
        #time 1 onward
        for t in range(1,T):
            s,x = sHist[t],xprime[sHist[t]]
            c,n,xprime = np.empty(self.S),nf[s](x),np.empty(self.S)
            for shat in range(self.S):
                c[shat] = cf[shat](x)
            for sprime in range(self.S):
                xprime[sprime] = xprimef[s,sprime](x)
                
            Tau = self.Tau(c,n)[s]
            u_c = Uc(c,n)
            Eu_c = Pi[sHist[t-1]].dot(u_c)
            muHist[t] = self.Vf[s](x,1)
            
            RHist[t-1] = Uc(cHist[t-1],nHist[t-1])/(self.beta*Eu_c)
            
            cHist[t],nHist[t],Bhist[t],TauHist[t] = c[s],n,x/u_c[s],Tau
        
        return cHist,nHist,Bhist,TauHist,sHist,muHist,RHist
    
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
        for s in range(self.S):
            for x in xgrid:
                xprime0 = np.empty(self.S)
                for sprime in range(self.S):
                    xprime0[sprime] = xprimef[s,sprime](x)
                self.z0[x,s] = np.hstack([cf[s](x),nf[s](x),xprime0])
                
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
            self.zFB[s] = np.hstack([self.cFB[s],self.nFB[s],self.xFB])
                
            
        
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
    
    def get_policies_time1(self,x,s,Vf):
        '''
        Finds the optimal policies 
        '''
        Para,beta,Theta,G,S,Pi = self.Para,self.beta,self.Theta,self.G,self.S,self.Pi
        U,Uc,Un = Para.U,Para.Uc,Para.Un
        
        def objf(z):
            c,n,xprime = z[0],z[1],z[2:]
            Vprime = np.empty(S)
            for sprime in range(S):
                Vprime[sprime] = Vf[sprime](xprime[sprime])

            return -(U(c,n)+beta*Pi[s].dot(Vprime))
            
        def cons(z):
            c,n,xprime = z[0],z[1],z[2:]
            return np.hstack([
            x - Uc(c,n)*c-Un(c,n)*n - beta*Pi[s].dot(xprime),
            (Theta*n - c - G)[s]            
            ])
            
        
        out,fx,_,imode,smode = fmin_slsqp(objf,self.z0[x,s],f_eqcons=cons,
                            bounds=[(0.,100),(0.,100)]+[self.xbar]*S,full_output=True,iprint=0)
                            
        if imode >0:
            raise Exception(smode)
           
        self.z0[x,s] = out
        return np.hstack([-fx,out])
        
    def get_policies_time0(self,B_,s0,Vf):
        '''
        Finds the optimal policies 
        '''
        Para,beta,Theta,G,S,Pi = self.Para,self.beta,self.Theta,self.G,self.S,self.Pi
        U,Uc,Un = Para.U,Para.Uc,Para.Un
        
        def objf(z):
            c,n,xprime = z[0],z[1],z[2:]
            Vprime = np.empty(S)
            for sprime in range(S):
                Vprime[sprime] = Vf[sprime](xprime[sprime])

            return -(U(c,n)+beta*Pi[s0].dot(Vprime))
            
        def cons(z):
            c,n,xprime = z[0],z[1],z[2:]
            return np.hstack([
            -Uc(c,n)*(c-B_)-Un(c,n)*n - beta*Pi[s0].dot(xprime),
            (Theta*n - c - G)[s0]            
            ])
            
        
        out,fx,_,imode,smode = fmin_slsqp(objf,self.zFB[s0],f_eqcons=cons,
                            bounds=[(0.,100),(0.,100)]+[self.xbar]*S,full_output=True,iprint=0)
                            
        if imode >0:
            raise Exception(smode)
           
        return np.hstack([-fx,out])