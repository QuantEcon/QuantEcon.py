from matplotlib import pyplot as plt
import numpy as np
from quantecon import ConsumerProblem
from quantecon import compute_fixed_point
from scipy import interp
from quantecon import mc_tools 

def compute_asset_series(cp, T=500000):
    """
    Simulates a time series of length T for assets, given optimal savings
    behavior.  Parameter cp is an instance of consumerProblem
    """

    Pi, z_vals, R = cp.Pi, cp.z_vals, cp.R  # Simplify names
    v_init, c_init = cp.initialize()
    c = compute_fixed_point(cp.coleman_operator, c_init)
    cf = lambda a, i_z: interp(a, cp.asset_grid, c[:, i_z])
    a = np.zeros(T+1)
    z_seq = mc_tools.mc_sample_path(Pi, sample_size=T)
    for t in range(T):
        i_z = z_seq[t]
        a[t+1] = R * a[t] + z_vals[i_z] - cf(a[t], i_z)
    return a

if __name__ == '__main__':

    cp = ConsumerProblem(r=0.03, grid_max=4)
    a = compute_asset_series(cp)
    fig, ax = plt.subplots()
    ax.hist(a, bins=20, alpha=0.5, normed=True)
    ax.set_xlabel('assets')
    ax.set_xlim(-0.05, 0.75)
    plt.show()
