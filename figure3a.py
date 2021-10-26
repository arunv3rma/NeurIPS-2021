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



# Batching based algorithm with Control Variate
def bacthing(mu, omega, sigma_m, sigma_w, T, batch_size=10):
    K                   = len(mu)           # Number of arms
    batch_rewards       = np.zeros(K)       # Collected rewards of batch for arms
    batch_cv            = np.zeros(K)       # Collected CV of batch for arms
    batch_counter       = np.zeros(K)       # Keep count of observations in batch
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
    
    # Maximum mean reward
    max_mean_reward = max(np.exp(mu + (sigma_m**2)/2))

    # Stores instantaneous regret of each round
    instantaneous_regret = []               

    # Initialization: Sampling each arm once
    for s in range(3*K):
        # Samples
        k = s % K
        for _ in range(batch_size):
            random_sample = np.random.lognormal(mean=mu[k], sigma=sigma_m[k], size=None)
            arm_cv_value = np.random.lognormal(mean=omega[k], sigma=sigma_w[k], size=None)
            arm_reward = random_sample + arm_cv_value
            batch_rewards[k] += arm_reward
            batch_cv[k] += arm_cv_value

            # Regret
            instantaneous_regret.append(max_mean_reward - np.exp(mu[k] + (sigma_m[k]**2)/2))
        
        # Update all variables
        batch_mean_reward = batch_rewards[k]/batch_size
        batch_mean_cv = batch_cv[k]/batch_size
        arm_rewards[k] += batch_mean_reward
        arm_rewards_seq[k] += (batch_mean_reward**2)
        arm_cv[k] += batch_mean_cv
        arm_cv_seq[k] += (batch_mean_cv**2)
        arm_cross_terms[k] += (batch_mean_reward*batch_mean_cv)
        num_pulls[k] += 1
        batch_rewards[k] = 0
        batch_cv[k] = 0

    # Remaining Rounds
    for t in range(3*batch_size*K, T):
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
        random_sample = np.random.lognormal(mean=mu[I_t], sigma=sigma_m[I_t], size=None)
        arm_cv_value = np.random.lognormal(mean=omega[I_t], sigma=sigma_w[I_t], size=None)
        arm_reward = random_sample + arm_cv_value
        batch_rewards[I_t] += arm_reward
        batch_cv[I_t] += arm_cv_value
        batch_counter[I_t] += 1

        # Update all variables
        if batch_counter[I_t] == batch_size:
            batch_mean_reward = batch_rewards[I_t]/batch_size
            batch_mean_cv = batch_cv[I_t]/batch_size
            arm_rewards[I_t] += batch_mean_reward
            arm_rewards_seq[I_t] += (batch_mean_reward**2)
            arm_cv[I_t] += batch_mean_cv
            arm_cv_seq[I_t] += (batch_mean_cv**2)
            arm_cross_terms[I_t] += (batch_mean_reward*batch_mean_cv)
            num_pulls[I_t] += 1
            batch_rewards[I_t] = 0
            batch_cv[I_t] = 0
            batch_counter[I_t] = 0

        # Regret
        instantaneous_regret.append(max_mean_reward - np.exp(mu[I_t] + (sigma_m[I_t]**2)/2))
     
    # Returning instantaneous regret       
    return instantaneous_regret



# Jackknifing based algorithm with Control Variate
def jackknifing(mu, omega, sigma_m, sigma_w, T):
    K                   = len(mu)           # Number of arms
    full_rewards        = np.zeros(K)       # All rewards for arms
    full_cv             = np.zeros(K)       # All Control variates for arms
    full_rewards_seq    = np.zeros(K)       # Sequare of all rewards for arms
    full_cv_seq         = np.zeros(K)       # Sequare of all control variates for arms
    full_cross_terms    = np.zeros(K)       # Product of all rewards and CV for arms
    mu_cv_est           = np.zeros(K)       # Estimated mean of new estimator for arms
    arm_rewards         = np.zeros((K, T))  # Collected rewards for arms
    arm_rewards_seq     = np.zeros((K, T))  # Collected sequare of rewards for arms
    arm_cv              = np.zeros((K, T))  # Collected CV for arms
    arm_cv_seq          = np.zeros((K, T))  # Collected sequare of CV for arms
    arm_cross_terms     = np.zeros((K, T))  # Collected product of reward and CV for arms
    num_pulls           = np.zeros(K, int)  # Number of arm pulls
    mu_hat              = np.zeros(K)       # Estimated mean rewards of arms
    var_hat             = np.zeros(K)       # Estimated variance of arms's rewards
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = []               

    # Initialization: Sampling each arm once
    for s in range(2*K):
        # Samples
        k = s % K
        random_sample = np.random.normal(mu[k], sigma_m[k], size=None)
        arm_cv_value = np.random.normal(omega[k], sigma_w[k], size=None)
        arm_reward = random_sample + arm_cv_value
        
        # Updating reward and CV values variables
        full_rewards[k] += arm_reward
        full_cv[k] += arm_cv_value
        full_rewards_seq[k] += (arm_reward**2)
        full_cv_seq[k] += (arm_cv_value**2)
        full_cross_terms[k] += (arm_reward*arm_cv_value)

        # Subtracting the observation from sum
        arm_rewards[k] += arm_reward
        arm_rewards[k, num_pulls[k]] -= arm_reward
        arm_rewards_seq[k] += (arm_reward**2)
        arm_rewards_seq[k, num_pulls[k]] -= (arm_reward**2)
        arm_cv[k] += arm_cv_value
        arm_cv[k, num_pulls[k]] -= arm_cv_value
        arm_cv_seq[k] += (arm_cv_value**2)
        arm_cv_seq[k, num_pulls[k]] -= (arm_cv_value**2)
        arm_cross_terms[k] += (arm_reward*arm_cv_value)
        arm_cross_terms[k, num_pulls[k]] -= (arm_reward*arm_cv_value)
        num_pulls[k] += 1

        # Regret
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])
        
    # ### Computing means with full observations ###
    # Estimated mean rewards of arms
    full_mu_est = full_rewards/num_pulls
    full_cv_est = full_cv/num_pulls

    # Computing sequare of centered cv values of arms
    full_cv_centered_seq = full_cv_seq + (num_pulls*(omega**2)) - (2.0*omega*full_cv)

    # Computing beta value
    beta = (full_cross_terms - (omega*full_rewards) - (full_mu_est*full_cv) + (num_pulls*full_mu_est*omega))/full_cv_centered_seq

    # Estimated mean of new estimator
    mu_cv_est = full_mu_est + (beta*(omega - full_cv_est))

    # Getting new observations via Splitting method
    for k in range(K):
        s = num_pulls[k] - 1

        # Estimated mean rewards of arms
        mu_est = arm_rewards[k, :s+1]/s
        cv_est = arm_cv[k, :s+1]/s
        
        # Computing sequare of centered cv values of arms
        cv_centered_seq = arm_cv_seq[k, :s+1] + ((s+1)*(omega[k]**2)) - (2.0*omega[k]*s*cv_est)
        
        # Computing beta value
        beta = (arm_cross_terms[k, :s+1] - (omega[k]*s*mu_est) - (mu_est*s*cv_est) + (omega[k]*(s+1)*mu_est))/cv_centered_seq
        
        # Getting new observations
        new_observations = ((s+1)*mu_cv_est[k]) - (s*(mu_est + (beta*omega[k]) - (beta*cv_est)))

        # Getting mean and variance (scaled by 1/s)
        mu_hat[k] = np.mean(new_observations)
        var_hat[k] = np.var(new_observations)/s

    # Remaining Rounds
    for t in range(2*K, T):
        # Calculating the UCBs for each arm
        arm_ucb = mu_hat + (ss.t.ppf(1.0-(1.0/(t**2)), num_pulls-1)*np.sqrt(var_hat))

        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)
    
        # Samples
        random_sample = np.random.normal(mu[I_t], sigma_m[I_t], size=None)
        arm_cv_value = np.random.normal(omega[I_t], sigma_w[I_t], size=None)
        arm_reward = random_sample + arm_cv_value
        
        # Updating reward and CV values variables
        full_rewards[I_t] += arm_reward
        full_cv[I_t] += arm_cv_value
        full_rewards_seq[I_t] += (arm_reward**2)
        full_cv_seq[I_t] += (arm_cv_value**2)
        full_cross_terms[I_t] += (arm_reward*arm_cv_value)
        
        # Subtracting the observation from sum
        arm_rewards[I_t] += arm_reward
        arm_rewards[I_t, num_pulls[I_t]] -= arm_reward
        arm_rewards_seq[I_t] += (arm_reward**2)
        arm_rewards_seq[I_t, num_pulls[I_t]] -= (arm_reward**2)
        arm_cv[I_t] += arm_cv_value
        arm_cv[I_t, num_pulls[I_t]] -= arm_cv_value
        arm_cv_seq[I_t] += (arm_cv_value**2)
        arm_cv_seq[I_t, num_pulls[I_t]] -= (arm_cv_value**2)
        arm_cross_terms[I_t] += (arm_reward*arm_cv_value)
        arm_cross_terms[I_t, num_pulls[I_t]] -= (arm_reward*arm_cv_value)
        num_pulls[I_t] += 1

        # Regret
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])

        # ### Updates using all observations ###
        # Estimated mean rewards of arms
        full_mu_est = full_rewards[I_t]/num_pulls[I_t]
        full_cv_est = full_cv[I_t]/num_pulls[I_t]

        # Computing sequare of centered cv values of arms
        full_cv_centered_seq = full_cv_seq[I_t] + (num_pulls[I_t]*(omega[I_t]**2)) - (2.0*omega[I_t]*full_cv[I_t])

        # Computing beta value
        beta = (full_cross_terms[I_t] - (omega[I_t]*full_rewards[I_t]) - (full_mu_est*full_cv[I_t]) + (num_pulls[I_t]*full_mu_est*omega[I_t]))/full_cv_centered_seq

        # Estimated mean of new estimator
        mu_cv_est[I_t] = full_mu_est + (beta*(omega[I_t] - full_cv_est))

        # ### Updating each observations ###
        # Updating number of samples
        s = num_pulls[I_t] - 1

        # Estimated mean rewards of arms
        mu_est = arm_rewards[I_t, :s+1]/s
        cv_est = arm_cv[I_t, :s+1]/s
        
        # Computing sequare of centered cv values of arms
        cv_centered_seq = arm_cv_seq[I_t, :s+1] + ((s+1)*(omega[I_t]**2)) - (2.0*omega[I_t]*s*cv_est)
        
        # Computing beta value
        beta = (arm_cross_terms[I_t, :s+1] - (omega[I_t]*s*mu_est) - (mu_est*s*cv_est) + (omega[I_t]*(s+1)*mu_est))/cv_centered_seq
        
        # Getting new observations
        new_observations = ((s+1)*mu_cv_est[I_t]) - (s*(mu_est + (beta*(omega[I_t] - cv_est))))
        # if t == 400:
        #     print (np.mean(mu_cv_est[I_t]), mu[I_t], np.mean(mu_est + (beta*omega[I_t]) - (beta*cv_est)))
        #     print (np.mean((mu_est + (beta*(omega[I_t] -cv_est)))))
        #     print (np.mean(new_observations))
        # Getting mean and variance (scaled by 1/s)
        mu_hat[I_t] = np.mean(new_observations)
        var_hat[I_t] = np.var(new_observations)/s
     
    # Returning instantaneous regret       
    return instantaneous_regret


# Splitting based algorithm with Control Variate
def splitting(mu, omega, sigma_m, sigma_w, T):
    K                   = len(mu)           # Number of arms
    rewards             = np.zeros((K, T))  # Rewards for arms
    cv                  = np.zeros((K, T))  # Control variates for arms
    arm_rewards         = np.zeros((K, T))  # Collected rewards for arms
    arm_rewards_seq     = np.zeros((K, T))  # Collected sequare of rewards for arms
    arm_cv              = np.zeros((K, T))  # Collected CV for arms
    arm_cv_seq          = np.zeros((K, T))  # Collected sequare of CV for arms
    arm_cross_terms     = np.zeros((K, T))  # Collected product of reward and CV for arms
    num_pulls           = np.zeros(K, int)  # Number of arm pulls
    mu_hat              = np.zeros(K)       # Estimated mean rewards of arms
    var_hat             = np.zeros(K)       # Estimated variance of arms's rewards
    max_mean_reward     = max(mu + omega)   # Maximum mean reward

    # Stores instantaneous regret of each round
    instantaneous_regret = []               

    # Initialization: Sampling each arm once
    for s in range(2*K):
        # Samples
        k = s % K
        random_sample = np.random.normal(mu[k], sigma_m[k], size=None)
        arm_cv_value = np.random.normal(omega[k], sigma_w[k], size=None)
        arm_reward = random_sample + arm_cv_value
        
        # Updating reward and CV values variables
        rewards[k, num_pulls[k]] = arm_reward
        cv[k, num_pulls[k]] = arm_cv_value
        
        # Subtracting the observation from sum
        arm_rewards[k] += arm_reward
        arm_rewards[k, num_pulls[k]] -= arm_reward
        arm_rewards_seq[k] += (arm_reward**2)
        arm_rewards_seq[k, num_pulls[k]] -= (arm_reward**2)
        arm_cv[k] += arm_cv_value
        arm_cv[k, num_pulls[k]] -= arm_cv_value
        arm_cv_seq[k] += (arm_cv_value**2)
        arm_cv_seq[k, num_pulls[k]] -= (arm_cv_value**2)
        arm_cross_terms[k] += (arm_reward*arm_cv_value)
        arm_cross_terms[k, num_pulls[k]] -= (arm_reward*arm_cv_value)
        num_pulls[k] += 1

        # Regret
        instantaneous_regret.append(max_mean_reward - mu[k] - omega[k])
        
    # Getting new observations via Splitting method
    for k in range(K):
        s = num_pulls[k] - 1

        # Estimated mean rewards of arms
        mu_est = arm_rewards[k, :s+1]/s
        cv_est = arm_cv[k, :s+1]/s
        
        # Computing sequare of centered cv values of arms
        cv_centered_seq = arm_cv_seq[k, :s+1] + ((s+1)*(omega[k]**2)) - (2.0*omega[k]*s*cv_est)
        
        # Computing beta value
        beta = (arm_cross_terms[k, :s+1] - (omega[k]*s*mu_est) - (mu_est*s*cv_est) + (omega[k]*(s+1)*mu_est))/cv_centered_seq
        
        # Getting new observations
        new_observations = rewards[k, :s+1] + (beta*(omega[k] - cv[k, :s+1]))

        # Getting mean and variance (scaled by 1/s)
        mu_hat[k] = np.mean(new_observations)
        var_hat[k] = np.var(new_observations)/s

    # Remaining Rounds
    for t in range(2*K, T):
        # Calculating the UCBs for each arm
        arm_ucb = mu_hat + (ss.t.ppf(1.0-(1.0/(t**2)), num_pulls-1)*np.sqrt(var_hat))

        # Selecting arm with maximum UCB1 index value
        I_t = np.argmax(arm_ucb)

        # Samples
        random_sample = np.random.normal(mu[I_t], sigma_m[I_t], size=None)
        arm_cv_value = np.random.normal(omega[I_t], sigma_w[I_t], size=None)
        arm_reward = random_sample + arm_cv_value
        
        # Updating reward and CV values variables
        rewards[I_t, num_pulls[I_t]] = arm_reward
        cv[I_t, num_pulls[I_t]] = arm_cv_value
        
        # Subtracting the observation from sum
        arm_rewards[I_t] += arm_reward
        arm_rewards[I_t, num_pulls[I_t]] -= arm_reward
        arm_rewards_seq[I_t] += (arm_reward**2)
        arm_rewards_seq[I_t, num_pulls[I_t]] -= (arm_reward**2)
        arm_cv[I_t] += arm_cv_value
        arm_cv[I_t, num_pulls[I_t]] -= arm_cv_value
        arm_cv_seq[I_t] += (arm_cv_value**2)
        arm_cv_seq[I_t, num_pulls[I_t]] -= (arm_cv_value**2)
        arm_cross_terms[I_t] += (arm_reward*arm_cv_value)
        arm_cross_terms[I_t, num_pulls[I_t]] -= (arm_reward*arm_cv_value)
        num_pulls[I_t] += 1

        # Regret
        instantaneous_regret.append(max_mean_reward - mu[I_t] - omega[I_t])

        # Updating Mean and Variance
        s = num_pulls[I_t] - 1

        # Estimated mean rewards of arms
        mu_est = arm_rewards[I_t, :s+1]/s
        cv_est = arm_cv[I_t, :s+1]/s
        
        # Computing sequare of centered cv values of arms
        cv_centered_seq = arm_cv_seq[I_t, :s+1] + ((s+1)*(omega[I_t]**2)) - (2.0*omega[I_t]*s*cv_est)
        
        # Computing beta value
        beta = (arm_cross_terms[I_t, :s+1] - (omega[I_t]*s*mu_est) - (mu_est*s*cv_est) + (omega[I_t]*(s+1)*mu_est))/cv_centered_seq
        
        # Getting new observations
        new_observations = rewards[I_t, :s+1] + (beta*(omega[I_t] - cv[I_t, :s+1]))

        # Getting mean and variance (scaled by 1/s)
        mu_hat[I_t] = np.mean(new_observations)
        var_hat[I_t] = np.var(new_observations)/s

    # Returning instantaneous regret    
    return instantaneous_regret

# #######################################################################


# ############################## Main Code ##############################
# ######## Dataset details ########
samples = 5000
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
cases           = ['Jackknifing', 'Batching', 'UCB-CV', 'Splitting'] 
total_cases     = len(cases)

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
algos_regret    = []
for _ in tqdm(range(runs)):
    run_regret = []
    iter_regret = []
    for c in range(total_cases):
        if cases[c] == 'Batching':
            iter_regret = bacthing(arms_mean, cv_mean, arms_var, cv_var, samples)

        elif cases[c] == 'Jackknifing':
            iter_regret = jackknifing(arms_mean, cv_mean, arms_var, cv_var, samples)
        
        elif cases[c] == 'Splitting':
            iter_regret = splitting(arms_mean, cv_mean, arms_var, cv_var, samples)
        
        elif cases[c] == 'UCB-CV':
            iter_regret = ucb_cv(arms_mean, cv_mean, arms_var, cv_var, samples)

        run_regret.append(iter_regret)

    algos_regret.append(run_regret)

# ########## Plotting parameters ##########
xlabel              = "Rounds"
ylabel              = "Regret"
file_to_save        = "figure3a.png"
title               = "Comparison of Algorithms"
save_to_path        = "plots/" 
location_to_save    = save_to_path + file_to_save
plotting_parameters = [xlabel, ylabel, title, location_to_save, cases, samples]

# Regret Plotting
regret_plotting(algos_regret, total_cases, plotting_parameters)