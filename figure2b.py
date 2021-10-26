# !/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy.lib.shape_base import kron
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as ss
from scipy.optimize import bisect
from scipy.stats import entropy
from scipy.stats.morestats import probplot

from tqdm import tqdm


# ########################## Plotting functions #########################
# Getting Average regret and Confidence interval
def accumulative_regret_error(regret):
    time_horizon = [0]
    samples = len(regret[0])
    runs = len(regret)
    batch = samples / 10
    # batch = 40

    # Time horizon
    t = 0
    while True:
        t += 1
        if time_horizon[-1] + batch > samples:
            if time_horizon[-1] != samples:
                time_horizon.append(time_horizon[-1] + samples % batch)
            break
        time_horizon.append(time_horizon[-1] + batch)

    # Mean batch regret of R runs
    avg_batched_regret = []
    for r in range(runs):
        count = 0
        accumulative_regret = 0
        batch_regret = [0]
        for s in range(samples):
            count += 1
            accumulative_regret += regret[r][s]
            if count == batch:
                batch_regret.append(accumulative_regret)
                count = 0

        if samples % batch != 0:
            batch_regret.append(accumulative_regret)
        avg_batched_regret.append(batch_regret)

    regret = np.mean(avg_batched_regret, axis=0)

    # Confidence interval
    conf_regret = []
    freedom_degree = runs - 1
    for r in range(len(avg_batched_regret[0])):
        conf_regret.append(ss.t.ppf(0.95, freedom_degree) *
                           ss.sem(np.array(avg_batched_regret)[:, r]))
    return time_horizon, regret, conf_regret


# Regret Plotting
def regret_plotting(regret, cases, plotting_info):
    colors = list("gbcmryk")
    shape = ['--^', '--v', '--H', '--d', '--+', '--*']

    # Scatter Error bar with scatter plot
    for c in range(cases):
        horizon, batched_regret, error = accumulative_regret_error(np.array(regret)[:, c])
        plt.errorbar(horizon, batched_regret, error, color=colors[c])
        plt.plot(horizon, batched_regret, colors[c] + shape[c], label=plotting_info[4][c])

    plt.rc('font', size=10)                     # controls default text sizes
    # plt.title(plotting_info[2])
    plt.legend(loc='upper left', numpoints=1)   # Location of the legend
    plt.xlabel(plotting_info[0], fontsize=15)
    plt.ylabel(plotting_info[1], fontsize=15)

    # plt.axis([0, samples, -20, samples])
    # plt.xscale('log')
    plt.savefig(plotting_info[3], bbox_inches='tight')

    plt.close()
# #######################################################################


# #############################  Algorithms #############################
# UCB1 algorithm
def ucb1(mu, omega, sigma, sigma_w, T):
    K                   = len(mu)           # Number of arms
    arm_rewards         = np.zeros(K)       # Collected rewards for arms
    num_pulls           = np.zeros(K)       # Number of arm pulls
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = [] 

    # Initialization: Sampling each arm once
    for k in range(K):
        arm_rewards[k] += np.random.normal(mu[k], sigma[k], 1)[0] + np.random.normal(omega[k], sigma_w[k], 1)[0]
        num_pulls[k] += 1
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])

    # Remaining Rounds
    for t in range(K, T):
        # Calculating the UCBs for each arm
        arm_ucb = arm_rewards/num_pulls + np.sqrt((2.0*np.log(t))/num_pulls)

        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Updating variables
        arm_rewards[I_t] += np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        num_pulls[I_t] += 1
        instantaneous_regret.append(max_mean_reward - mu[I_t]-omega[I_t])
     
    # Returning instantaneous regret      
    return instantaneous_regret


# Thompson Sampling 
def ts(mu, omega, sigma, sigma_w, T):
    K                   = len(mu)           # Number of arms
    arm_rewards         = np.zeros(K)       # Collected rewards for arms
    num_pulls           = np.ones(K)        # Number of arm pulls
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Storing Instantaneous regret
    instantaneous_regret = []

    # Remaining Rounds
    for t in range(T):
        # Sampling for each arm
        arm_theta = np.random.normal(arm_rewards/num_pulls, np.sqrt(1.0/num_pulls))
    
        # Selecting arm with maximum theta value
        I_t = np.argmax(arm_theta)
       
        # Updating variables
        arm_rewards[I_t] += np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        num_pulls[I_t] += 1
        instantaneous_regret.append(max_mean_reward - mu[I_t]-omega[I_t])

    # Returning instantaneous regret
    return instantaneous_regret


# UCB-V algorithm
def ucb_v(mu, omega, sigma, sigma_w, T):
    K                   = len(mu)           # Number of arms
    arm_rewards         = np.zeros(K)       # Collected rewards for arms
    arm_rewards_seq     = np.zeros(K)       # Collected sequare of rewards for arms
    num_pulls           = np.zeros(K)       # Number of arm pulls
    mu_est              = np.zeros(K)       # Estimated mean rewards of arms
    sample_var          = np.zeros(K)       # Sample variance of arms
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = []               

    # Initialization: Sampling each arm once
    for k in range(K):
        arm_reward = np.random.normal(mu[k], sigma[k], 1)[0] + np.random.normal(omega[k], sigma_w[k], 1)[0]
        arm_rewards[k] += arm_reward
        arm_rewards_seq[k] += (arm_reward**2)
        num_pulls[k] += 1
        instantaneous_regret.append(max_mean_reward - mu[k]-omega[k])

    # Remaining Rounds
    for t in range(K, T):
        # Estimated mean rewards
        mu_est = arm_rewards/num_pulls

        # Calculating variance
        sample_var = (arm_rewards_seq/num_pulls) - (mu_est**2)

        # Calculating the UCBs for each arm
        arm_ucb = mu_est + np.sqrt((2.2*np.log(t)*sample_var)/num_pulls) + ((3.3*np.log(t))/num_pulls)

        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Updating variables
        arm_reward = np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        arm_rewards[I_t] += arm_reward
        arm_rewards_seq[I_t] += (arm_reward**2)
        num_pulls[I_t] += 1
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])
     
    # Returning instantaneous regret       
    return instantaneous_regret


# UCB-V algorithm
def eucbv(mu, omega, sigma, sigma_w, T):
    K                   = len(mu)           # Number of arms
    arm_rewards         = np.zeros(K)       # Collected rewards for arms
    arm_rewards_seq     = np.zeros(K)       # Collected sequare of rewards for arms
    num_pulls           = np.zeros(K)       # Number of arm pulls
    mu_est              = np.zeros(K)       # Estimated mean rewards of arms
    sample_var          = np.zeros(K)       # Sample variance of arms
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = []               

    # Initialization: Sampling each arm once
    for k in range(K):
        arm_reward = np.random.normal(mu[k], sigma[k], 1)[0] + np.random.normal(omega[k], sigma_w[k], 1)[0]
        arm_rewards[k] += arm_reward
        arm_rewards_seq[k] += (arm_reward**2)
        num_pulls[k] += 1
        instantaneous_regret.append(max_mean_reward - mu[k]-omega[k])

    # Remaining Rounds
    for t in range(K, T):
        # Estimated mean rewards
        mu_est = arm_rewards/num_pulls

        # Calculating variance
        sample_var = (arm_rewards_seq/num_pulls) - (mu_est**2)

        # Calculating the UCBs for each arm
        arm_ucb = mu_est + np.sqrt( ((sample_var + 2.0) * np.log(T/K)) / (4.0*num_pulls) )

        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Updating variables
        arm_reward = np.random.normal(mu[I_t], sigma[I_t], 1)[0] + np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        arm_rewards[I_t] += arm_reward
        arm_rewards_seq[I_t] += (arm_reward**2)
        num_pulls[I_t] += 1
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])
     
    # Returning instantaneous regret       
    return instantaneous_regret


# UCB based algorithm with Control Variate
def ucb_cv(mu, omega, sigma, sigma_w, T):
    K                   = len(mu)           # Number of arms
    arm_rewards         = np.zeros(K)       # Collected rewards for arms
    arm_rewards_seq     = np.zeros(K)       # Collected sequare of rewards for arms
    arm_cv              = np.zeros(K)       # Collected CV for arms
    arm_cv_seq          = np.zeros(K)       # Collected sequare of CV for arms
    arm_cross_terms     = np.zeros(K)       # Collected product of reward and CV for arms
    num_pulls           = np.zeros(K)       # Number of arm pulls
    mu_est              = np.zeros(K)       # Estimated mean rewards of arms
    cv_est              = np.zeros(K)       # Estimated CV of arms
    cv_centered_seq     = np.zeros(K)       # Estimated sum of sequare of centered cv values of arms
    beta                = np.zeros(K)       # Beta value of arm
    mu_cv_est           = np.zeros(K)       # Estimated mean of new estimator for arms
    cv_rewards_seq      = np.zeros(K)       # Estimated sum of new observations of arms
    sample_var          = np.zeros(K)       # Estimated sample variance of arms
    var_mult            = np.zeros(K)       # Multiplier to get variance of estimator
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = []               

    # Initialization: Sampling each arm once
    for k in range(3*K):
        # Samples
        k = k % K
        random_sample = np.random.normal(mu[k], sigma[k], 1)[0]
        arm_cv_value = np.random.normal(omega[k], sigma_w[k], 1)[0]
        arm_reward = random_sample + arm_cv_value
        
        # Update all variables
        arm_rewards[k] += arm_reward
        arm_rewards_seq[k] += (arm_reward**2)
        arm_cv[k] += arm_cv_value
        arm_cv_seq[k] += (arm_cv_value**2)
        arm_cross_terms[k] += (arm_reward*arm_cv_value)
        num_pulls[k] += 1
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])

    # Remaining Rounds
    for t in range(3*K, T):
        # Estimated mean rewards of arms
        mu_est = arm_rewards/num_pulls
        cv_est = arm_cv/num_pulls

        # Computing sequare of centered cv values of arms
        cv_centered_seq = arm_cv_seq + (num_pulls*(omega**2)) - (2.0*omega*arm_cv)

        # Computing beta value
        beta = (arm_cross_terms - (omega*arm_rewards) - (mu_est*arm_cv) + (num_pulls*mu_est*omega))/cv_centered_seq

        # Estimated mean of new estimator
        mu_cv_est = mu_est + (beta*omega) - (beta*cv_est)

        # Computing sum of sequare of new observation
        cv_rewards_seq = arm_rewards_seq + (beta*beta*cv_centered_seq) - (2*beta*arm_cross_terms) + (2*beta*omega*arm_rewards)

        # Computing sample variance of arms
        sample_var = (1.0/(num_pulls - 2)) * (cv_rewards_seq - (num_pulls*mu_cv_est*mu_cv_est) ) # - (2*mu_cv_est*arm_rewards) )

        # Multiplier of sample variance to get variance of new estimator
        var_mult = (1.0/num_pulls) / (1.0 - ( ((arm_cv - (num_pulls*omega))**2) / (num_pulls*cv_centered_seq)) )

        # Calculating the UCBs for each arm
        arm_ucb = mu_cv_est + (ss.t.ppf(1.0-(1.0/(t**2)), num_pulls-2)*np.sqrt(var_mult*sample_var))

        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Samples
        random_sample = np.random.normal(mu[I_t], sigma[I_t], 1)[0]
        arm_cv_value = np.random.normal(omega[I_t], sigma_w[I_t], 1)[0]
        arm_reward = random_sample + arm_cv_value

        # Update all variables
        arm_rewards[I_t] += arm_reward
        arm_rewards_seq[I_t] += (arm_reward**2)
        arm_cv[I_t] += arm_cv_value
        arm_cv_seq[I_t] += (arm_cv_value**2)
        arm_cross_terms[I_t] += (arm_reward*arm_cv_value)
        num_pulls[I_t] += 1

        # Regret
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])
     
    # Returning instantaneous regret       
    return instantaneous_regret

# #######################################################################


# ############################## Main Code ##############################
# ######## Dataset details ########
samples = 50000
runs    = 100
np.random.seed(100)

# ######## Problem Instances ########
arms            = 10            
max_arm_mean    = 0.06*arms
max_cv_mean     = 0.08*arms
max_arm_var     = 0.1
max_cv_var      = 0.1
arm_gap         = 0.05
cv_gap          = 0.05

# Data parameters
generate_data       = True      # True: Generate new data, False: Used saved generated data
save_data           = False     # True: Save new generated data, False: Do not save data

# Mean vector
arms_mean   = np.zeros(arms)
arms_var    = np.zeros(arms)
cv_mean     = np.zeros(arms)
cv_var      = np.zeros(arms)
for k in range(arms):
    arms_mean[k] = max_arm_mean - (k*arm_gap)
    arms_var[k] = max_arm_var
    cv_mean[k] = max_cv_mean - (k*cv_gap)
    cv_var[k] = max_arm_var

# Runnging algorithm
cases           = ['UCB1', 'UCB-V', 'TS', 'EUCBV', 'UCB-CV']#
total_cases     = len(cases)
algos_regret    = []
for _ in tqdm(range(runs)):
    run_regret = []
    iter_regret = []
    for c in range(total_cases):
        if cases[c] == 'UCB1':
            iter_regret = ucb1(arms_mean, cv_mean, arms_var, cv_var, samples)

        elif cases[c] == 'UCB-V':
            iter_regret = ucb_v(arms_mean, cv_mean, arms_var, cv_var, samples)
        
        elif cases[c] == 'EUCBV':
            iter_regret = eucbv(arms_mean, cv_mean, arms_var, cv_var, samples)

        elif cases[c] == 'TS':
            iter_regret = ts(arms_mean, cv_mean, arms_var, cv_var, samples)
        
        elif cases[c] == 'UCB-CV':
            iter_regret = ucb_cv(arms_mean, cv_mean, arms_var, cv_var, samples)

        run_regret.append(iter_regret)

    algos_regret.append(run_regret)

# ########## Plotting parameters ##########
xlabel              = "Rounds"
ylabel              = "Regret"
file_to_save        = "figure2b.png"
title               = "Comparison of Algorithms"
save_to_path        = "plots/" 
location_to_save    = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases, samples]

# Regret Plotting
regret_plotting(algos_regret, total_cases, plotting_parameters)