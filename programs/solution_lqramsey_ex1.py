
import numpy as np
from numpy import array
from lqramsey import *

# == Parameters == #
beta = 1 / 1.05   
rho, mg = .95, .35
A = array([[0, 0, 0, rho, mg*(1-rho)],
           [1, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1]])
C = np.zeros((5, 1))
C[0, 0] = np.sqrt(1 - rho**2) * mg / 8
Sg = array((1, 0, 0, 0, 0)).reshape(1, 5)        
Sd = array((0, 0, 0, 0, 0)).reshape(1, 5)       
Sb = array((0, 0, 0, 0, 2.135)).reshape(1, 5)  # Chosen st. (Sc + Sg) * x0 = 1
Ss = array((0, 0, 0, 0, 0)).reshape(1, 5)

economy = Economy(beta=beta, 
        Sg=Sg, 
        Sd=Sd, 
        Sb=Sb, 
        Ss=Ss, 
        discrete=False, 
        proc=(A, C))

T = 50
path = compute_paths(T, economy)
gen_fig_1(path)


