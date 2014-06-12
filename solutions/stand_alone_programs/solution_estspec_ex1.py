
import numpy as np
import matplotlib.pyplot as plt
# from quantecon.linproc import linearProcess
# from quantecon.estspec import periodogram
import quantecon as qe


## Data
n = 400
phi = 0.5
theta = 0, -0.8
lp = qe.linearProcess(phi, theta)
X = lp.simulation(ts_length=n)

fig, ax = plt.subplots(3, 1)

for i, wl in enumerate((15, 55, 175)):  # window lengths
    
    x, y = qe.periodogram(X)
    ax[i].plot(x, y, 'b-', lw=2, alpha=0.5, label='periodogram')

    x_sd, y_sd = lp.spectral_density(two_pi=False, resolution=120)
    ax[i].plot(x_sd, y_sd, 'r-', lw=2, alpha=0.8, label='spectral density')

    x, y_smoothed = qe.periodogram(X, window='hamming', window_len=wl)
    ax[i].plot(x, y_smoothed, 'k-', lw=2, label='smoothed periodogram')

    ax[i].legend()
    ax[i].set_title('window length = {}'.format(wl))

plt.show()

