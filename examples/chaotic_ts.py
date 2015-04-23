"""
Filename: choatic_ts.py
Reference: http://quant-econ.net/py/python_oop.html
"""
from chaos_class import Chaos
import matplotlib.pyplot as plt

ch = Chaos(0.1, 4.0) 
ts_length = 250

fig, ax = plt.subplots()
ax.set_xlabel(r'$t$', fontsize=14)
ax.set_ylabel(r'$x_t$', fontsize=14)
x = ch.generate_sequence(ts_length)
ax.plot(range(ts_length), x, 'bo-', alpha=0.5, lw=2, label=r'$x_t$')
plt.show()
