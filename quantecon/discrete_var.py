from quantecon import matrix_eqn as qme
import numpy as np
from numpy.random import multivariate_normal
from numpy.linalg import matrix_power

def tpm(A, 
        Omega,
        nvec=None,
        upper_bounds=None,
        seed=1234,
        sim_length=1_000_000,
        burn_in=100_000):
    
    """
    This code discretizes a VAR(1) process of the form:

        x_t = A * x_t-1 + Omega * e_t
    
    For a mathematical derivation check *Finite-State Approximation Of
    VAR Processes:  A Simulation Approach* by Stephanie Schmitt-Grohé and
    Martín Uribe, July 11, 2010. 
    
    This code was adapted by Carlos Rondón-Moreno from Schmitt-Grohé and
    Uribe's code for MATLAB.
     
    Inputs:
    -   A is an m x m matrix containing the process' autocorrelation parameters
    -   Omega is an m x m variance - covariance matrix
    -   nvec is either None or an m-vector containing the number of grid
        points in the discretization of each element of x_t. If nvec is None,
        then nvec is set to (10, ..., 10).
    -   sim_length is the the length of the simulated time series. By default,
        sim_length=1_000_000
    -   burn_in is the number of burn-in draws from the simulated series. By
        default, burn_in = 100_000
    -   upper_bounds is an m-vector indicating the upper bound value of the
        grid. The grid will be simmetrical around 0. If upper_bounds is None,
        then it will be reset to sqrt(10)*std(x_t(i)), evaluated at the
        stationary distribution.
    
    Outputs:
    -   Pi is an squared-matrix containing the transition probability
        matrix of the discretized state. By default, the code removes 
        the states that are never visited. 
    -   S is a matrix where the element (i,j) of S is the discretized
        value of the j-th element of x_t in state i. Reducing S to its
        unique values yields the grid values.
    -   Xvec is a matrix of size m by sim_length containing the simulated time
        series of the m discretized states.    


    Notes:
    -   The code presently assumes normal shocks but normality is not required
        for the algorithm to work. The draws from the multivariate standard
        normal generator can be replaced by any other random number generator
        with mean 0 and unit standard deviation.
    
    
    # Example:

    -   This example discretizes the stohcastic process used to calibrate
        the economic model included in ``Downward Nominal Wage Rigidity, 
        Currency Pegs, and Involuntary Unemployment'' by Stephanie
        Schmitt-Grohé and Martín Uribe, Journal of Political Economy 124,
        October 2016, 1466-1514. 
            
            A     = np.array([[0.7901, -1.3570], [-0.0104, 0.8638]])
            Omega = np.array([[0.0012346, -0.0000776], [-0.0000776, 0.0000401]])
            nvec = np.array([21, 11])
            Pi, Xvec, S = tpm(A, Omega, nvec, 
                              sim_length=1_000_000, burn_in = 100_000)
    """
        
    np.random.seed(seed)
    m, r = len(A), len(Omega)
    
    if nvec is None:
        nvec = np.full(m, 10)
    
    if upper_bounds is None:
        # Compute stationary variance-covariance matrix of AR process and use
        # it to obtain grid bounds.
        Sigma = qme.solve_discrete_lyapunov(A, Omega) 
        sigma_vector = np.sqrt(np.diagonal(Sigma))    # Stationary std dev
        upper_bounds = np.sqrt(10) * sigma_vector
    
    V = []
    for i in range(m):
        b = np.linspace(-upper_bounds[i], upper_bounds[i], nvec[i])
        V.append(b)

    n = nvec.prod()     # Total number of possible values of the discretized state
    S = np.zeros([n, m])
    
    for i in range(m):
        if i == 0:
            temp    = np.ravel(V[i])
            S[:, i] = np.ravel(np.tile(temp,[np.prod(nvec[i+1:]), 1] ))
        else:
            temp = np.sort(np.ravel(np.tile(V[i],[np.prod(nvec[0:i]), 1])))
            S[:, i] = np.ravel(np.tile(temp,[np.prod(nvec[i+1:]), 1] ))

    Pi = np.zeros((n, n))
    x0 = np.zeros((m, 1))
    xx = np.zeros((n, m))
    d = np.sum((S - xx)**2, axis=1)
    ind_i = np.argmin(d)
    Xvec = np.zeros((m, sim_length))
    mean = np.zeros(m)
    
    # Run simulation to compute transition probabilities
    for t in range(sim_length + burn_in):
        drw = multivariate_normal(mean, Omega).reshape(m, 1)
        x = A @ x0 + drw
        
        xx = np.tile(x.T, (n, 1))
        d = np.sum((S - xx)**2, axis=1)
        ind_j = np.argmin(d)

        if t > burn_in:
            Pi[ind_i, ind_j] += 1
            Xvec[:, t-burn_in] = x.T
        x0 = x
        ind_i = ind_j
        
        if np.mod(t, 100_000)==0:
             print(t)
    
    # Cut states where the column sum of Pi is zero (i.e., inaccesible states
    # according to the simulation)
    indx = np.where(np.sum(Pi, axis=0) > 0)
    Pi = Pi[indx[0], :]
    Pi = Pi[:, indx[0]]
    S  = S[indx[0], :]
    
    # Normalize
    sum_row = np.sum(Pi, axis=1)
    for i in range(len(Pi)):
        Pi[i,:] = Pi[i,:] / sum_row[i]
    
    return Pi, Xvec, S



if __name__ == '__main__':
    T = 10_000
    burn_in = 1_000
    A = np.array([[0.7901, -1.3570], 
                  [-0.0104, 0.8638]])
    Omega = np.array([[0.0012346, -0.0000776], 
                      [-0.0000776, 0.0000401]])
    nvec = np.array((3, 4))
    Pi, Xvec, S = tpm(A, Omega, nvec, sim_length=T, burn_in=burn_in)
    print(S)
    print(Pi)

