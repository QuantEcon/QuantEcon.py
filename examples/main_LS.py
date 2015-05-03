# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:07:56 2015

@author: dgevans
"""
import matplotlib.pyplot as plt
import numpy as np
import lucas_stokey as LS
from calibrations.BGP import M1
from calibrations.CES import M2
from calibrations.CES import M_time_example


'''
Time Varying Example
'''

PP_seq_time = LS.Planners_Allocation_Sequential(M_time_example) #solve sequential problem

sHist_h = np.array([0,1,2,3,5,5,5])
sHist_l = np.array([0,1,2,4,5,5,5])

sim_seq_h = PP_seq_time.simulate(1.,0,7,sHist_h)
sim_seq_l = PP_seq_time.simulate(1.,0,7,sHist_l)

plt.figure(figsize=[14,10])
plt.subplot(3,2,1)
plt.title('Consumption')
plt.plot(sim_seq_l[0],'-ok')
plt.plot(sim_seq_h[0],'-or')
plt.subplot(3,2,2)
plt.title('Labor Supply')
plt.plot(sim_seq_l[1],'-ok')
plt.plot(sim_seq_h[1],'-or')
plt.subplot(3,2,3)
plt.title('Government Debt')
plt.plot(sim_seq_l[2],'-ok')
plt.plot(sim_seq_h[2],'-or')
plt.subplot(3,2,4)
plt.title('Taxe Rate')
plt.plot(sim_seq_l[3],'-ok')
plt.plot(sim_seq_h[3],'-or')
plt.subplot(3,2,5)
plt.title('Government Spending')
plt.plot(M_time_example.G[sHist_l],'-ok')
plt.plot(M_time_example.G[sHist_h],'-or')
plt.subplot(3,2,6)
plt.title('Output')
plt.plot(M_time_example.Theta[sHist_l]*sim_seq_l[1],'-ok')
plt.plot(M_time_example.Theta[sHist_h]*sim_seq_h[1],'-or')

plt.tight_layout()
plt.savefig('TaxSequence_time_varying.png')

plt.figure(figsize=[8,5])
plt.title('Gross Interest Rate')
plt.plot(sim_seq_l[-1],'-ok')
plt.plot(sim_seq_h[-1],'-or')
plt.tight_layout()
plt.savefig('InterestRate_time_varying.png')

'''
Time 0 example
'''
PP_seq_time0 = LS.Planners_Allocation_Sequential(M2) #solve sequential problem

B_vec = np.linspace(-1.5,1.,100)
taxpolicy = np.vstack([PP_seq_time0.simulate(B_,0,2)[3] for B_ in B_vec])
interest_rate = np.vstack([PP_seq_time0.simulate(B_,0,3)[-1] for B_ in B_vec])

plt.figure(figsize=[14,6])
plt.subplot(211)
plt.plot(B_vec,taxpolicy[:,0],linewidth=2.)
plt.plot(B_vec,taxpolicy[:,1],linewidth=2.)

plt.title('Tax Rate')
plt.legend((r'Time $t=0$', 'Time $t\geq1$'),loc=2,shadow=True)
plt.subplot(212)
plt.title('Gross Interest Rate')
plt.plot(B_vec,interest_rate[:,0],linewidth=2.)
plt.plot(B_vec,interest_rate[:,1],linewidth=2.)
plt.xlabel('Initial Government Debt')
plt.tight_layout()

plt.savefig('Time0_taxpolicy.png')




#compute the debt entered with at time 1
B1_vec = np.hstack([PP_seq_time0.simulate(B_,0,2)[2][1] for B_ in B_vec])
#now compute the optimal policy if the government could reset
tau1_reset = np.hstack([PP_seq_time0.simulate(B1,0,1)[3] for B1 in B1_vec])

plt.figure(figsize=[10,6])
plt.plot(B_vec,taxpolicy[:,1],linewidth=2.)
plt.plot(B_vec,tau1_reset,linewidth=2.)
plt.xlabel('Initial Government Debt')
plt.title('Tax Rate')
plt.legend((r'$\tau_1$', r'$\tau_1^R$'),loc=2,shadow=True)
plt.tight_layout()

plt.savefig('Time0_inconsistent.png')


'''
BGP Example
'''
#initialize mugrid for value function iteration
muvec = np.linspace(-0.6,0.0,200)


PP_seq = LS.Planners_Allocation_Sequential(M1) #solve sequential problem
PP_bel = LS.Planners_Allocation_Bellman(M1,muvec) #solve recursive problem

T = 20
#sHist = utilities.simulate_markov(M1.Pi,0,T)
sHist = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],dtype=int)

#simulate
sim_seq = PP_seq.simulate(0.5,0,T,sHist)
sim_bel = PP_bel.simulate(0.5,0,T,sHist)

#plot policies
plt.figure(figsize=[14,10])
plt.subplot(3,2,1)
plt.title('Consumption')
plt.plot(sim_seq[0],'-ok')
plt.plot(sim_bel[0],'-xk')
plt.legend(('Sequential','Recursive'),loc='best')
plt.subplot(3,2,2)
plt.title('Labor Supply')
plt.plot(sim_seq[1],'-ok')
plt.plot(sim_bel[1],'-xk')
plt.subplot(3,2,3)
plt.title('Government Debt')
plt.plot(sim_seq[2],'-ok')
plt.plot(sim_bel[2],'-xk')
plt.subplot(3,2,4)
plt.title('Tax Rate')
plt.plot(sim_seq[3],'-ok')
plt.plot(sim_bel[3],'-xk')
plt.subplot(3,2,5)
plt.title('Government Spending')
plt.plot(M1.G[sHist],'-ok')
plt.plot(M1.G[sHist],'-xk')
plt.plot(M1.G[sHist],'-^k')
plt.subplot(3,2,6)
plt.title('Output')
plt.plot(M1.Theta[sHist]*sim_seq[1],'-ok')
plt.plot(M1.Theta[sHist]*sim_bel[1],'-xk')

plt.tight_layout()
plt.savefig('TaxSequence_LS.png')

plt.figure(figsize=[8,5])
plt.title('Gross Interest Rate')
plt.plot(sim_seq[-1],'-ok')
plt.plot(sim_bel[-1],'-xk')
plt.legend(('Sequential','Recursive'),loc='best')
plt.tight_layout()
