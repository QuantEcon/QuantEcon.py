# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 10:47:42 2015

@author: dgevans
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

class interpolate_wrapper(object):
    '''
    Wrapper to interpolate vector function
    '''
    def __init__(self,F):
        '''
        Inits with array of interpolated functions
        '''
        self.F = F
        
    def __getitem__(self,index):
        '''
        Uses square brakets operator
        '''
        return interpolate_wrapper(np.asarray(self.F[index]))
        
        
    def reshape(self,*args):
        '''
        Reshapes F
        '''
        self.F = self.F.reshape(*args)
        return self
        
    def transpose(self):
        '''
        Transpose F
        '''
        self.F = self.F.transpose()
        
    def __len__(self):
        '''
        return length
        '''
        return len(self.F)
        
    def __call__(self,xvec):
        '''
        Evaluates F at X for each element of F, keeping track of the shape of F
        '''
        x = np.atleast_1d(xvec)
        shape = self.F.shape
        if len(x) == 1:
            fhat = np.hstack([f(x) for f in self.F.flatten()])
            return fhat.reshape(shape)
        else:
            fhat = np.vstack([f(x) for f in self.F.flatten()])
            return fhat.reshape( np.hstack((shape,len(x))) )

class interpolator_factory(object):
    '''
    Generates an interpolator factory which will interpolate vector functions
    '''
    def __init__(self,k,s):
        '''
        Inits with types, orders and k
        '''
        self.k = k
        self.s = s
        
    def __call__(self,xgrid,Fs):
        '''
        Interpolates function given function values Fs at domain X
        '''
        shape,m = Fs.shape[:-1],Fs.shape[-1]
        Fs = Fs.reshape((-1,m))
        F = []
        for Fhat in Fs:
            #F.append(interpolate(X,Fs[:,i],self.INFO))
            F.append(UnivariateSpline(xgrid,Fhat,k=self.k,s=self.s))
        return interpolate_wrapper(np.array(F).reshape(shape))
        
        
def fun_vstack(fun_list):
    '''
    Performs vstack on interpolator wrapper
    '''
    Fs = [IW.F for IW in fun_list]
    return interpolate_wrapper(np.vstack(Fs))
    
def fun_hstack(fun_list):
    '''
    Performs vstack on interpolator wrapper
    '''
    Fs = [IW.F for IW in fun_list]
    return interpolate_wrapper(np.hstack(Fs))
    
def simulate_markov(Pi,s_0,T):  
    '''
    Simulates markov chain Pi for T periods starting at s_0
    '''   
    
    sHist = np.empty(T,dtype = int)
    sHist[0] = s_0
    S = len(Pi)
    for t in range(1,T):
        sHist[t] = np.random.choice(np.arange(S),p=Pi[sHist[t-1]])
        
    return sHist