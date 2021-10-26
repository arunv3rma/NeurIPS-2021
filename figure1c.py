# !/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Data Range
T = 100000
s = np.array(range(50, 200, 1))

# Compute different confidence interval
t_value_1q = (stats.t.ppf(1.0-(1.0/(T**2)), s-1-1) / (stats.t.ppf(1.0-(1.0/(T**2)), T-1-1)))**1
t_value_2q = (stats.t.ppf(1.0-(1.0/((1*T)**2)), s-2-1) / (stats.t.ppf(1.0-(1.0/(T**2)), T-2-1)))**1
t_value_3q = (stats.t.ppf(1.0-(1.0/((1*T)**2)), s-3-1) / (stats.t.ppf(1.0-(1.0/(T**2)), T-3-1)))**1
t_value_5q = (stats.t.ppf(1.0-(1.0/((1*T)**2)), s-5-1) / (stats.t.ppf(1.0-(1.0/(T**2)), T-5-1)))**1
t_value_10q = (stats.t.ppf(1.0-(1.0/((1*T)**2)), s-10-1) / (stats.t.ppf(1.0-(1.0/(T**2)), T-10-1)))**1
t_value_20q = (stats.t.ppf(1.0-(1.0/((1*T)**2)), s-20-1) / (stats.t.ppf(1.0-(1.0/(T**2)), T-20-1)))**1

# Plotting Data
colors = list("gbcmryk")
shape = ['--^', '--v', '--H', '--d', '--+', '--*']

# colors[1] + shape[1]
plt.plot(s, t_value_1q, colors[1], label=r'$q=1$')
plt.plot(s, t_value_2q, colors[2], label=r'$q=2$')
plt.plot(s, t_value_3q, colors[3], label=r'$q=3$')
plt.plot(s, t_value_5q, colors[4], label=r'$q=5$')
plt.plot(s, t_value_10q, colors[5], label=r'$q=10$')
plt.plot(s, t_value_20q, colors[6], label=r'$q=20$')


plt.rc('font', size=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20) 
plt.legend(loc='upper right', numpoints=1)

plt.xlabel(r'$N_i(T)$', fontsize=20)
plt.ylabel(r'$\frac{V_{T,N_i(T),q}^{(2)}}{V_{T,T,q}^{(2)}}$ where $N_{i}(T) > 50$', fontsize=20)
plt.savefig('plots/figure1c.png', bbox_inches='tight')