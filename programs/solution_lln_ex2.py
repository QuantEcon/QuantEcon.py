"""
Illustrates a consequence of the vector CLT.  The underlying random vector is
X = (W, U + W), where W is Uniform(-1, 1), U is Uniform(-2, 2), and U and W
are independent of each other.
"""
import numpy as np
from scipy.stats import uniform, chi2
from scipy.linalg import inv, sqrtm
import matplotlib.pyplot as plt

# == Set parameters == #
n = 250
replications = 50000
dw = uniform(loc=-1, scale=2)  # Uniform(-1, 1)
du = uniform(loc=-2, scale=4)  # Uniform(-2, 2)
sw, su = dw.std(), du.std()
vw, vu = sw**2, su**2
Sigma = ((vw, vw), (vw, vw + vu))
Sigma = np.array(Sigma)

# == Compute Sigma^{-1/2} == #
Q = inv(sqrtm(Sigma))  

# == Generate observations of the normalized sample mean == #
error_obs = np.empty((2, replications))
for i in range(replications):
    # == Generate one sequence of bivariate shocks == #
    X = np.empty((2, n))
    W = dw.rvs(n)
    U = du.rvs(n)
    # == Construct the n observations of the random vector == #
    X[0, :] = W
    X[1, :] = W + U
    # == Construct the i-th observation of Y_n == #
    error_obs[:, i] = np.sqrt(n) * X.mean(axis=1)

# == Premultiply by Q and then take the squared norm == #
temp = np.dot(Q, error_obs)
chisq_obs = np.sum(temp**2, axis=0)

# == Plot == #
plt, ax = plt.subplots()
xmax = 8
ax.set_xlim(0, 8)
xgrid = np.linspace(0, 8, 200)
lb = "Chi-squared with 2 degrees of freedom"
ax.plot(xgrid, chi2.pdf(xgrid, 2), 'k-', lw=2, label=lb)
ax.legend()
ax.hist(chisq_obs, bins=50, normed=True)

plt.show()
