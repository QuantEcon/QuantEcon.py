# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 14:07:56 2015

@author: dgevans
"""
import matplotlib.pyplot as plt
import numpy as np
import lucas_stokey as LS
import amss
from calibrations.BGP import M1
from calibrations.CES import M1 as M_convergence
from calibrations.CES import M_time_example
import utilities

#initialize mugrid for value function iteration
muvec = np.linspace(-0.7,0.01,200)


'''
Time Varying Example
'''

M_time_example.transfers = True #Government can use transfers
PP_seq_time = LS.Planners_Allocation_Sequential(M_time_example) #solve sequential problem
PP_im_time = amss.Planners_Allocation_Bellman(M_time_example,muvec)

sHist_h = np.array([0,1,2,3,5,5,5])
sHist_l = np.array([0,1,2,4,5,5,5])

sim_seq_h = PP_seq_time.simulate(1.,0,7,sHist_h)
sim_im_h = PP_im_time.simulate(1.,0,7,sHist_h)
sim_seq_l = PP_seq_time.simulate(1.,0,7,sHist_l)
sim_im_l = PP_im_time.simulate(1.,0,7,sHist_l)

plt.figure(figsize=[14,10])
plt.subplot(3,2,1)
plt.title('Consumption')
plt.plot(sim_seq_l[0],'-ok')
plt.plot(sim_im_l[0],'-or')
plt.plot(sim_seq_h[0],'-^k')
plt.plot(sim_im_h[0],'-^r')
plt.subplot(3,2,2)
plt.title('Labor')
plt.plot(sim_seq_l[1],'-ok')
plt.plot(sim_im_l[1],'-or')
plt.plot(sim_seq_h[1],'-^k')
plt.plot(sim_im_h[1],'-^r')
plt.legend(('Complete Markets','Incomplete Markets'),loc='best')
plt.subplot(3,2,3)
plt.title('Government Debt')
plt.plot(sim_seq_l[2],'-ok')
plt.plot(sim_im_l[2],'-or')
plt.plot(sim_seq_h[2],'-^k')
plt.plot(sim_im_h[2],'-^r')
plt.subplot(3,2,4)
plt.title('Tax Rate')
plt.plot(sim_seq_l[3],'-ok')
plt.plot(sim_im_l[4],'-or')
plt.plot(sim_seq_h[3],'-^k')
plt.plot(sim_im_h[4],'-^r')
plt.subplot(3,2,5)
plt.title('Government Spending')
plt.plot(M_time_example.G[sHist_l],'-ok')
plt.plot(M_time_example.G[sHist_l],'-or')
plt.plot(M_time_example.G[sHist_h],'-^k')
plt.plot(M_time_example.G[sHist_h],'-^r')
plt.ylim([0.05,0.25])
plt.subplot(3,2,6)
plt.title('Output')
plt.plot(M_time_example.Theta[sHist_l]*sim_seq_l[1],'-ok')
plt.plot(M_time_example.Theta[sHist_l]*sim_im_l[1],'-or')
plt.plot(M_time_example.Theta[sHist_h]*sim_seq_h[1],'-^k')
plt.plot(M_time_example.Theta[sHist_h]*sim_im_h[1],'-^r')
plt.tight_layout()
plt.savefig('TaxSequence_time_varying_AMSS.png')




'''
BGP Example
'''

M1.transfers = False #Government can use transfers
PP_seq = LS.Planners_Allocation_Sequential(M1) #solve sequential problem
PP_bel = LS.Planners_Allocation_Bellman(M1,muvec) #solve recursive problem
PP_im = amss.Planners_Allocation_Bellman(M1,muvec)

T = 20
#sHist = utilities.simulate_markov(M1.Pi,0,T)
sHist = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],dtype=int)

#simulate
sim_seq = PP_seq.simulate(0.5,0,T,sHist)
#sim_bel = PP_bel.simulate(0.5,0,T,sHist)
sim_im = PP_im.simulate(0.5,0,T,sHist)

#plot policies
plt.figure(figsize=[14,10])
plt.subplot(3,2,1)
plt.title('Consumption')
plt.plot(sim_seq[0],'-ok')
#plt.plot(sim_bel[0],'-xk')
plt.plot(sim_im[0],'-^k')
plt.legend(('Complete Markets','Incomplete Markets'),loc='best')
plt.subplot(3,2,2)
plt.title('Labor')
plt.plot(sim_seq[1],'-ok')
#plt.plot(sim_bel[1],'-xk')
plt.plot(sim_im[1],'-^k')
plt.subplot(3,2,3)
plt.title('Government Debt')
plt.plot(sim_seq[2],'-ok')
#plt.plot(sim_bel[2],'-xk')
plt.plot(sim_im[2],'-^k')
plt.subplot(3,2,4)
plt.title('Tax Rate')
plt.plot(sim_seq[3],'-ok')
#plt.plot(sim_bel[3],'-xk')
plt.plot(sim_im[4],'-^k')
plt.subplot(3,2,5)
plt.title('Government Spending')
plt.plot(M1.G[sHist],'-ok')
#plt.plot(M1.G[sHist],'-^k')
plt.ylim([0.05,0.25])
plt.subplot(3,2,6)
plt.title('Output')
plt.plot(M1.Theta[sHist]*sim_seq[1],'-ok')
#plt.plot(M1.Theta[sHist]*sim_bel[1],'-xk')
plt.plot(M1.Theta[sHist]*sim_im[1],'-^k')
plt.savefig('TaxSequence_AMSS.png')
plt.tight_layout()


#Now long simulations
T_long = 200
sim_seq_long = PP_seq.simulate(0.5,0.,T_long)
sHist_long = sim_seq_long[-3]
sim_im_long = PP_im.simulate(0.5,0.,T_long,sHist_long)

plt.figure(figsize=[14,10])
plt.subplot(3,2,1)
plt.title('Consumption')
plt.plot(sim_seq_long[0],'-k')
plt.plot(sim_im_long[0],'-.k')
plt.legend(('Complete Markets','Incomplete Markets'),loc='best')
plt.subplot(3,2,2)
plt.title('Labor')
plt.plot(sim_seq_long[1],'-k')
plt.plot(sim_im_long[1],'-.k')
plt.subplot(3,2,3)
plt.title('Government Debt')
plt.plot(sim_seq_long[2],'-k')
plt.plot(sim_im_long[2],'-.k')
plt.subplot(3,2,4)
plt.title('Tax Rate')
plt.plot(sim_seq_long[3],'-k')
plt.plot(sim_im_long[4],'-.k')
plt.subplot(3,2,5)
plt.title('Government Spending')
plt.plot(M1.G[sHist_long],'-k')
plt.plot(M1.G[sHist_long],'-.k')
plt.ylim([0.05,0.25])
plt.subplot(3,2,6)
plt.title('Output')
plt.plot(M1.Theta[sHist_long]*sim_seq_long[1],'-k')
plt.plot(M1.Theta[sHist_long]*sim_im_long[1],'-.k')
plt.tight_layout()
plt.savefig('Long_SimulationAMSS.png')

'''
Show Convergence example
'''
muvec = np.linspace(-0.15,0.0,100) #change 
PP_C = amss.Planners_Allocation_Bellman(M_convergence,muvec)
xgrid = PP_C.xgrid
xf = PP_C.policies[-2] #get x policies
plt.figure()
for s in range(2):
    plt.plot(xgrid,xf[0,s](xgrid)-xgrid)
    
sim_seq_convergence = PP_C.simulate(0.5,0.,2000)
sHist_long = sim_seq_convergence[-1]

plt.figure(figsize=[14,10])
plt.subplot(3,2,1)
plt.title('Consumption')
plt.plot(sim_seq_convergence[0],'-k')
plt.legend(('Complete Markets','Incomplete Markets'),loc='best')
plt.subplot(3,2,2)
plt.title('Labor')
plt.plot(sim_seq_convergence[1],'-k')
plt.subplot(3,2,3)
plt.title('Government Debt')
plt.plot(sim_seq_convergence[2],'-k')
plt.subplot(3,2,4)
plt.title('Tax Rate')
plt.plot(sim_seq_convergence[3],'-k')
plt.subplot(3,2,5)
plt.title('Government Spending')
plt.plot(M_convergence.G[sHist_long],'-k')
plt.ylim([0.05,0.25])
plt.subplot(3,2,6)
plt.title('Output')
plt.plot(M_convergence.Theta[sHist_long]*sim_seq_convergence[1],'-k')
plt.tight_layout()
plt.savefig('Convergence_SimulationAMSS.png')


