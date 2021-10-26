# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data Range
n = np.array(range(50, 100000000, 1000))

# Compute different confidence interval
t_value_1q = (stats.t.ppf(1.0-(1.0/(n**2)), n-1-1))**2 
t_value_2q = (stats.t.ppf(1.0-(1.0/(n**2)), n-2-1))**2
t_value_5q = (stats.t.ppf(1.0-(1.0/(n**2)), n-5-1))**2
t_value_10q = (stats.t.ppf(1.0-(1.0/(n**2)), n-10-1))**2
t_value_20q = (stats.t.ppf(1.0-(1.0/(n**2)), n-20-1))**2
ucb_cw = 3.726*np.log(n)

# Plotting Data
colors = list("gbcmryk")
shape = ['--^', '--v', '--H', '--d', '--+', '--*']
plt.plot(n, t_value_1q, colors[1], label=r'$q=1$')
plt.plot(n, t_value_2q, colors[2], label=r'$q=2$')
plt.plot(n, t_value_5q, colors[3], label=r'$q=5$')
plt.plot(n, t_value_10q, colors[4], label=r'$q=10$')
plt.plot(n, t_value_20q, colors[5], label=r'$q=20$')
plt.plot(n, ucb_cw, colors[0], label=r'$3.726*\log(T)$')

plt.rc('font', size=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc='lower right', numpoints=1)

plt.xlabel('T', fontsize=20)
plt.ylabel(r'$(V_{T,T,q}^{(2)})^2$ where $T>50$', fontsize=20)

# Saving the result
plt.savefig('plots/figure1b.png', bbox_inches='tight')
