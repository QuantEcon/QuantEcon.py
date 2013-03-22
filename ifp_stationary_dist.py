from matplotlib import pyplot as plt
import numpy as np
from ifp import consumerProblem, coleman_operator, initialize
from compute_fp import compute_fixed_point
from scipy import interp
import mc_sample 

def compute_asset_series(m, T=500000):
    """
    Simulates a time series of length T for assets, given optimal savings
    behavior.  Parameter m is an instance of consumerProblem
    """

    Pi, z_vals, R = m.Pi, m.z_vals, m.R  # Simplify names
    v_init, c_init = initialize(m)
    c = compute_fixed_point(coleman_operator, m, c_init)
    cf = lambda a, i_z: interp(a, m.asset_grid, c[:, i_z])
    a = np.zeros(T+1)
    z_seq = mc_sample.sample_path(Pi, sample_size=T)
    for t in range(T):
        i_z = z_seq[t]
        a[t+1] = R * a[t] + z_vals[i_z] - cf(a[t], i_z)
    return a

if __name__ == '__main__':

    m = consumerProblem(r=0.03, grid_max=4)
    a = compute_asset_series(m)
    fig, ax = plt.subplots()
    ax.hist(a, bins=20, alpha=0.5, normed=True)
    ax.set_xlabel('assets')
    ax.set_xlim(-0.05, 0.75)
    fig.show()
