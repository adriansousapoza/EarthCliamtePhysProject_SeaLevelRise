#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:23:58 2022

@author: asp
"""

import os
import numpy as np
import array
from scipy.stats import norm, lognorm
import emcee
import pandas as pd

datafolder = os.path.abspath('/home/asp/Downloads/Earth and Climate Physics/Project/data')

def read_csv_file(file):
    data = pd.read_csv(f"{datafolder}/{file}").copy()
    
    ref = 0
    for i in range(len(data['time'])):
        data['time'][i] = data['time'][i]//100 + (data['time'][i] % 100)*1/12
    
    y = np.array(data['gm_tas'])[1:]
    x = np.array(data['time'])[1:]
    y_mean = np.zeros(len(y[6:-6]))
    x_mean = x[6:-6]
    for i in range(len(y_mean)):
        y_mean[i] = np.mean(y[i:i+12])
        if np.isclose(data['time'][i], 2000):
            ref = y_mean[i]
    
    y_mean = y_mean - ref
    y = y - ref
    
    return x,y,x_mean,y_mean

def read_dat_file(file, historical = True):
    run_to_plot = 1
    if historical:
        yr_strt = 1801
        yr_end = 2040
        yr_ref = 2000
    else:
        yr_strt = 2001
        yr_end = 2310
        yr_ref = 2020
    
    n_mnths = 12 * (yr_end - yr_strt + 1) 
    n_ref = 12 * (yr_ref - yr_strt) + 1
    
    fid = open(f"{datafolder}/{file}", mode = 'rb')
    a = array.array("i") 
    a.fromfile(fid, 1)
    b = a[0]
    n_vals = b // 4
    n_runs = n_vals // n_mnths
    
    tmp = array.array("f")
    tmp.fromfile(fid, b//4)
    tmp = np.asarray(tmp)
    strh_gm = np.reshape(tmp, (n_runs, n_mnths), 'F')
    strh_gm[strh_gm<-1.7e7] = "NaN"
    
    for i in range(n_runs):
        strh_gm[i,:] = strh_gm[i,:] - strh_gm[i,n_ref]
    
    x = np.zeros(n_mnths)
    for i in range(n_mnths):
        x[i] = (i + 0.5) / 12 + yr_strt
    
    y = strh_gm[run_to_plot-1,:][~np.isnan(strh_gm[run_to_plot-1,:])]
    x = x[~np.isnan(strh_gm[run_to_plot-1,:])]
    
    y_mean = np.zeros(len(y[6:-6]))
    x_mean = x[6:-6]
    for i in range(0,len(y_mean)):
        y_mean[i] = np.mean(y[i:i+12])
    
    return x,y,x_mean,y_mean

def steric_model(tas_timeseries, a, b, tau, S0):
    delta_t=1
    S = np.empty_like(tas_timeseries)
    S[0] = S0
    for i in range(1,len(tas_timeseries)):
        Seq = a*(tas_timeseries[i]) + b
        dSdt = (Seq-S[i-1])/tau
        S[i] = S[i-1] + dSdt*delta_t
    return S


def log_likelihood(theta): 
    a, b, tau, S0, sigma_SL = theta 
    model = steric_model(temperature, a,b,tau,S0) 
    return np.sum(norm.logpdf(y_mean, loc=model, scale=sigma_SL)) 

def log_prior(theta): 
    a, b, tau, S0, sigma = theta 
    logp = lognorm.logpdf(sigma/0.03, s=0.5)
    logp = logp + lognorm.logpdf(tau/100, s=0.3)
    if np.isinf(logp) or np.isnan(logp):
        return -np.inf
    return logp

def log_post(theta): 
    logp = log_prior(theta)
    if np.isinf(logp):
        return logp
    logp = logp + log_likelihood(theta)
    return logp


file = "raw_data/CMIP6 steric SSH/cmip6_CMIP_historical_strh_zostoga_gm.dat"
x,y,x_mean,y_mean = read_dat_file(file)

file = "raw_data/cmip6_tas_for_steric_analysis/historical/gm_tas_CMIP_historical_ACCESS-CM2_r1i1p1f1_gn.csv"
x2,y2,x_mean2,temperature = read_csv_file(file)

runs = 5000
pos = [2.2e-01, 1.3e-01, 1.2e+03, -6.4e-02, 3.3e-03] + 1e-4 * np.random.randn(32, 5)
pos = np.float64(pos)
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_post)
sampler.run_mcmc(pos, runs, progress=True)
tau = sampler.get_autocorr_time()
print(tau)

samples = sampler.get_chain(flat = True)
flat_samples = sampler.get_chain(discard=1000, flat=True)

print(np.mean(flat_samples,axis=0))

np.savetxt("samples_dat_{}runs.csv".format(runs), samples, delimiter=",")
np.savetxt("samples_dat_disc_{}runs.csv".format(runs), flat_samples, delimiter=",")








