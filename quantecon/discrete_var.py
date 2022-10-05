
import numpy as np
from numpy.random import multivariate_normal
from numpy.linalg import matrix_power

def tpm(A,                 # A is a m X m matrix containing the autocorrelation parameters
        omega,             # Variance - Covariance Matrix
        N = float("nan"),  # Number of grid points
        UB = float("nan"), # Upper Bound
        T=1_000_000,       # Length of the simulated series
        Tburn=100_000):
    
    """
        This code discretizes a VAR(1) process of the form:
        x_t = A * x_t-1 + omega * e_t
        
        For a mathematical derivation check ``Finite-State Approximation Of  VAR Processes:  A Simulation Approach'' by 
        Stephanie Schmitt-Grohé and Martín Uribe, July 11, 2010. 
        
        This code was adapted by Carlos Rondón-Moreno from Schmitt-Grohé and Uribe's code for MATLAB.
         
        General Considerations:
        -   Normality is not required. The draws from the multivariate normal generator can be replaced by any other random number
            generator with mean 0 and unit standard deviation.
        
        Inputs:
        -   A is an m x m matrix containing the process' autocorrelation parameters
        -   Omega is an m x m variance - covariance matrix
        -   N is an m x 1 vector containing the number of grid points in the discretization of each element of x_t, By default, N = 10
        -   T is the the length of the simulated time series. By default, T = 1_000_000
        -   Tburn is the number of burn-in draws from the simulated series. By default, Tburn = 100_000
        -   UB is an m x 1 vector indicating the upper bound value of the grid. The grid will be simmetrical around 0. By default, 
            UB = sqrt(10)*std(x_t(i))
        
        Outputs:
        -   Pi is an squared-matrix containing the transition probability matrix of the discretized state. By default, the code removes 
            the states that are never visited. 
        -   S is a matrix where the element (i,j) of S is the discretized value of the j-th element of x_t in state i. Reducing S to its
            unique values yields the grid values.
        -   Xvec is a matrix of size m by T containing the simulated time series of the m discretized states.    
        
        Ancillary Functions:
        -   Mom function: computes the VAR(1)'s unconditional variance.
        
        # Example:

        -   This example discretizes  the sthocastic process used to calibrate the economic model included in ``Downward Nominal Wage Rigidity, 
            Currency Pegs, and Involuntary Unemployment'' by Stephanie Schmitt-Grohé and Martín Uribe, Journal of Political Economy 
            124, October 2016, 1466-1514. 
                
                A     = np.array([[0.7901, -1.3570], [-0.0104, 0.8638]])
                omega = np.array([[0.0012346, -0.0000776], [-0.0000776, 0.0000401]])
                N = np.array([21, 11])
                Pii, Xvec, S = tpm(A, omega, N, T=1_000_000, Tburn = 100_000)

        
    """
        
    m = len(A)
    r = len(omega)
    
    if np.isnan(np.sum(N)):
        N = 10 * np.ones(m, dtype = int)
    
    Sigg = mom(np.eye(m), A, omega)
    sigg = np.sqrt(np.diagonal(Sigg))  # Unconditional  standard deviation  of AR process
    
    
    if np.isnan(np.sum(UB)):
        UB = np.sqrt(10) * sigg
    
    V = []
    for i in range(m):
        b = np.linspace(-UB[i], UB[i], N[i])
        V.append(b)

    n = N.prod()     # Total number of possible values of the discretized state
    S = np.zeros([n, m])
    
    for i in range(m):
        if i == 0:
            temp    = np.ravel(V[i])
            S[:, i] = np.ravel(np.tile(temp,[np.prod(N[i+1:]), 1] ))
        else:
            temp = np.sort(np.ravel(np.tile(V[i],[np.prod(N[0:i]), 1])))
            S[:, i] = np.ravel(np.tile(temp,[np.prod(N[i+1:]), 1] ))

    Pi = np.zeros([n, n])
    x0 = np.zeros([m, 1])
    xx = np.zeros([n, m])
    d = np.sum((S-xx)**2, 1)
    ind_i = np.argmin(d)
    Xvec = np.zeros([m, T])
    mean = [0] * m
    
    for t in range(T+Tburn):
        drw = multivariate_normal(mean, omega).reshape(1,2)
        x =A @ x0 + drw.T
        
        xx = np.tile(x.T,[n,1])
        d = np.sum((S-xx)**2, 1)
        ind_j = np.argmin(d)

        if t > Tburn:
            Pi[ind_i,ind_j] = Pi[ind_i, ind_j] + 1
            Xvec[:,t-Tburn] = x.T
        
        x0 = x
        ind_i = ind_j
        
        if np.mod(t,100_000)==0:
             print(t)
    
    indx = np.where(np.sum(Pi,0)>0)
    Pi = Pi[indx[0], :]
    Pi = Pi[:,indx[0]]
    S  = S[indx[0], :]
    
    sum_row = np.sum(Pi,1)
    for i in range(len(Pi)):
        Pi[i,:] = Pi[i,:]/sum_row[i]
    
    return Pi, Xvec, S


def mom(gx, hx, varshock, J=0, method=True):
    hx_old = hx
    sig_old = varshock
    if method==True:
        hx_old = hx
        sig_old = varshock
        sigx_old = np.identity(len(hx))
        diff = 0.1
        tol = 1e-25
        while diff > tol:
            sigx = hx_old @ sigx_old @ hx_old.T + sig_old
            diff = np.max(np.max(np.abs(sigx - sigx_old)))
            sig_old= hx_old @ sig_old @ hx_old.T + sig_old
            hx_old= hx_old @ hx_old
            sigx_old = sigx
    else:
        F = np.kron(hx, hx)
        sigx = np.linalg.inv(np.identity(len(F)) - F) @ np.ravel(varshock)
        sigx = sigx.reshape(len(hx),len(hx))

    sigxJ = matrix_power(hx, -(np.min([0,J]))) @ sigx @ matrix_power(hx.T, (np.max([0,J])))
    sigyJ = np.real(gx @ sigxJ @ gx.T) 
    
    return sigyJ

############### Example:

T = 1_000_000
Tburn = 100_000
A     = np.array([[0.7901, -1.3570], [-0.0104, 0.8638]])
omega = np.array([[0.0012346, -0.0000776], [-0.0000776, 0.0000401]])
N = np.array([21, 11])
Pii, Xvec, S = tpm(A, omega, N, T=1_000_000, Tburn = 100_000)

