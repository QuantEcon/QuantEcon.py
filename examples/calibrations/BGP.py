# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 15:32:07 2015

@author: dgevans
"""
import numpy as np

class baseline(object):
    beta = 0.9
    psi = 0.69
    
    Pi = 0.5 *np.ones((2,2))
    
    G = np.array([0.1,0.2])
    
    Theta = np.ones(2)
    
    transfers = False
    
    #derivatives of utiltiy function
    def U(self,c,n):
        return np.log(c) + self.psi*np.log(1-n)
        
    def Uc(self,c,n):
        return 1./c
        
    def Ucc(self,c,n):
        return -c**(-2)
        
    def Un(self,c,n):
        return -self.psi/(1-n)
        
    def Unn(self,c,n):
        return -self.psi/(1-n)**2
        
        
#Model 1
M1 = baseline()

#Model 2

M2 = baseline()
M2.G = np.array([0.15])
M2.Pi = np.ones((1,1))
M2.Theta = np.ones(1)

#Model 3 with time varying

M_time_example = baseline()

M_time_example.Pi = np.array([[0., 1., 0.,   0.,  0., 0.],
                              [0., 0., 1.,   0.,  0., 0.],
                              [0., 0., 0., 0.5, 0.5,  0.],
                              [0., 0., 0., 0.,   0.,  1.],
                              [0., 0., 0., 0.,   0.,  1.],
                              [0., 0., 0., 0.,   0.,  1.]])
                         
M_time_example.G = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.1])
M_time_example.Theta = np.ones(6) # Theta can in principle be random