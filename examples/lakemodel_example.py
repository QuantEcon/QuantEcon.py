# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 18:08:44 2015

Author: David Evans

Example Usage of LakeModel.py
"""
import numpy as np
import matplotlib.pyplot as plt
import LakeModel

import pandas as pd
pd.set_option('display.mpl_style', 'default') # Make the graphs a bit prettier

#Initialize Parameters
alpha = 0.013
lamb = 0.283#0.2486
b = 0.0124
d = 0.00822
g = b-d
N0 = 150.
e0 = 0.92
u0 = 1-e0
T = 50

LM = LakeModel.LakeModel(lamb,alpha,b,d)

#Find steady state
xbar = LM.find_steady_state()

#simulate stocks for T periods
E0 = e0*N0
U0 = u0*N0
X_path = np.vstack( LM.simulate_stock_path([E0,U0],T) )
plt.figure(figsize=[10,9])
plt.subplot(3,1,1)
plt.plot(X_path[:,0])
plt.title(r'Employment')
plt.subplot(3,1,2)
plt.plot(X_path[:,1])
plt.title(r'Unemployment')
plt.subplot(3,1,3)
plt.plot(X_path.sum(1))
plt.title(r'Labor Force')
plt.tight_layout()
plt.savefig('example_stock_path.png')

#simulate rates for T periods

x_path = np.vstack( LM.simulate_rate_path([e0,u0],T) )
plt.figure(figsize=[10,6])
plt.subplot(2,1,1)
plt.plot(x_path[:,0])
plt.hlines(xbar[0],0,T,'r','--')
plt.title(r'Employment Rate')
plt.subplot(2,1,2)
plt.plot(x_path[:,1])
plt.hlines(xbar[1],0,T,'r','--')
plt.title(r'Unemployment Rate')
plt.tight_layout()
plt.savefig('example_rate_path.png')


#Simulate a single agent
T = 5000

A = LakeModel.LakeModelAgent(lamb,alpha)
pi_bar = A.compute_ergodic().flatten()

sHist = np.hstack(A.simulate(1,T))

pi_u = np.cumsum(sHist)/(np.arange(T) + 1.) # time spent in unemployment after T periods
pi_e = 1- pi_u #time spent employed

plt.figure(figsize=[10,6])
plt.subplot(2,1,1)
plt.plot(range(50,T),pi_e[50:])
plt.hlines(pi_bar[0],0,T,'r','--')
plt.title('Percent of Time Employed')
plt.subplot(2,1,2)
plt.plot(range(50,T),pi_u[50:])
plt.hlines(pi_bar[1],0,T,'r','--')
plt.xlabel('Time')
plt.title('Percent of Time Unemployed')
plt.tight_layout()
plt.savefig('example_averages.png')


#==============================================================================
# Now add McCall Search Model
#==============================================================================
from scipy.stats import norm

#using quaterly data
alpha_q = (1-(1-alpha)**3)
gamma = 1.

logw_dist = norm(np.log(20.),1)
w = np.linspace(0.,175,201)# wage grid

#compute probability of each wage level 
cdf = logw_dist.cdf(np.log(w))
pdf = cdf[1:]-cdf[:-1]
pdf /= pdf.sum()
w = (w[1:] + w[:1])/2

#Find the quilibirum
LME = LakeModel.LakeModel_Equilibrium(alpha_q,gamma,0.99,2.00,pdf,w)

#possible levels of unemployment insurance
cvec = np.linspace(1.,75,25)
T,W,U,EV,pi = map(np.vstack,zip(* [LME.find_steady_state_tax(c) for c in cvec]))
W= W[:]
T = T[:]
U = U[:]
EV = EV[:]
i_max = np.argmax(W)

plt.figure(figsize=[10,6])
plt.subplot(221)
plt.plot(cvec,W)
plt.xlabel(r'$c$')
plt.title(r'Welfare' )
axes = plt.gca()
plt.vlines(cvec[i_max],axes.get_ylim()[0],max(W),'k','-.')

plt.subplot(222)
plt.plot(cvec,T)
axes = plt.gca()
plt.vlines(cvec[i_max],axes.get_ylim()[0],T[i_max],'k','-.')
plt.xlabel(r'$c$')
plt.title(r'Taxes' )


plt.subplot(223)
plt.plot(cvec,pi[:,0])
axes = plt.gca()
plt.vlines(cvec[i_max],axes.get_ylim()[0],pi[i_max,0],'k','-.')
plt.xlabel(r'$c$')
plt.title(r'Employment Rate' )


plt.subplot(224)
plt.plot(cvec,pi[:,1])
axes = plt.gca()
plt.vlines(cvec[i_max],axes.get_ylim()[0],pi[i_max,1],'k','-.')
plt.xlabel(r'$c$')
plt.title(r'Unemployment Rate' )
plt.tight_layout()
plt.savefig('welfare_plot.png')