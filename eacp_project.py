#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:45:23 2022

@author: asp
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import array
from scipy.optimize import curve_fit
import corner
from sklearn.metrics import mean_squared_error as MSE

datafolder = os.path.abspath('/home/asp/Downloads/Earth and Climate Physics/Project/data')

def steric_model(tas_timeseries, a, b, tau, S0):
    delta_t=1
    S = np.empty_like(tas_timeseries)
    S[0] = S0
    for i in range(1,len(tas_timeseries)):
        Seq = a*(tas_timeseries[i]) + b
        dSdt = (Seq-S[i-1])/tau
        S[i] = S[i-1] + dSdt*delta_t
    return S

def read_csv_history(file):
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
    
    return x_mean,y_mean,ref

def read_csv_fut(file, ref):
    data = pd.read_csv(f"{datafolder}/{file}").copy()
    
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
        if x_mean[i] > 2100:
            y_mean = y_mean[:i]
            x_mean = x_mean[:i]
            break
        
    
    y_mean = y_mean - ref
    
    return x_mean, y_mean


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

def lin_func(x, a, c):
    return a*x + c

def poly_func(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def plot_data_mean(x,y,x_mean,y_mean):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(x, y, label = 'historical data')
    ax.plot(x_mean, y_mean, label = 'average over 12 months')
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.grid()
    ax.set_title("Global mean steric height (relative to Jan 2000)")
    ax.legend()
    fig.savefig('figures/hist_mean.pdf', dpi=400)
    plt.show()
    return True

def plot_mean_fit(x,y,y_fit, function):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(x, y, label = 'mean sea level')
    ax.plot(x, y_fit, label = function)
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.grid()
    ax.set_title("Global mean steric height (relative to Jan 2000)")
    ax.legend()
    fig.savefig('figures/{}.pdf'.format(function), dpi=400)
    plt.show()
    return True

def plot_samples(ndim, samples, labels):
    fig, axes = plt.subplots(ndim, dpi = 400, figsize=(10, 10), sharex=True)
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.grid()
    axes[-1].set_xlabel("step number")
    fig.savefig('figures/samples_convergence.pdf', dpi=400)
    plt.show()
    return True

def plot_check_emcee(flat_samples, size = 100):
    fig, ax = plt.subplots(dpi = 400)
    a,b,tau,S0 = flat_samples[-1,-1,:4]
    ax.plot(x_mean, steric_model(temperature, a, b, tau, S0), "C1", alpha=0.1, 
            label = 'last {} samples'.format(size))
    for i in range(size):
        a,b,tau,S0 = flat_samples[-i,0,:4]
        ax.plot(x_mean, steric_model(temperature, a, b, tau, S0), "C1", alpha=0.1)
    ax.plot(x_mean, y_mean, "k", label="historical data")
    ax.legend()
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.grid()
    ax.set_title("Global mean steric height (relative to Jan 2000)")
    fig.savefig('figures/check_emcee.pdf', dpi=400)
    plt.show()
    return True

# def plot_emcee_std():
#     fig, ax = plt.subplots(dpi = 400)
#     y1 = steric_model(temperature, mcmc[0,2], mcmc[1,2], mcmc[2,2], mcmc[3,2])
#     y2 = steric_model(temperature, mcmc[0,4], mcmc[1,4], mcmc[2,4], mcmc[3,4])
#     ax.plot(x_mean, steric_model(temperature, mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3]), "r", 
#             label = 'mean')
#     ax.plot(x_mean, steric_model(temperature, mcmc[0,2], mcmc[1,2], mcmc[2,2], mcmc[3,2]), "C1")
#     ax.plot(x_mean, steric_model(temperature, mcmc[0,4], mcmc[1,4], mcmc[2,4], mcmc[3,4]), "C1")
#     ax.fill_between(x_mean,y1,y2, color = "C1", alpha=.3, linewidth=0, label = '68.2%')
#     ax.plot(x_mean, y_mean, "k", label="historical")
#     ax.legend()
#     ax.set_xlabel("year")
#     ax.set_ylabel("metres")
#     ax.set_title("Global mean steric height (relative to Jan 2000)")
#     ax.grid()
#     fig.savefig('figures/plot_emcee_stdofvars.pdf', dpi=400)
#     plt.show()
#     return True

def plot_future(temp_fut, data):
    end = 2100
    x_fut = np.linspace(x_mean[-1], end, len(temp_fut))
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(x_fut, steric_model(np.concatenate((temperature,temp_fut),axis=None),
                                mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(temp_fut):],
            "r--", label = 'emcee (MLE)')
    ax.plot(x_mean, y_mean, "k", label="historical data")
    ax.plot(x_fut, steric_model(np.concatenate((temperature,temp_fut)), *popt)[-len(x_fut):], 
            "g--", label="scipy.optimize.curve_fit (LSE)")
    ax.grid()
    ax.legend()
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.set_title("Future gobal steric height (with data from {})".format(data))
    fig.savefig('figures/plot_future.pdf', dpi=400)
    plt.show()
    return True

def plot_compare_models(temp_fut, temp_fut2, temp_fut3, x_fut, x_hist):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(x_hist, y_mean, "k", label="historical")
    y_ssp126 = steric_model(np.concatenate((temperature,temp_fut),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp245 = steric_model(np.concatenate((temperature,temp_fut2),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp585 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    ax.plot(x_fut, y_ssp126, "C2", label="mean with data for ssp126")
    ax.plot(x_fut, y_ssp245, "C4", label="mean with data for ssp245")
    ax.plot(x_fut, y_ssp585, "C5", label="mean with data for ssp585")
    ax.plot(ssp126_x_mean, ssp126_y_mean, "C2--", label="ssp126")
    ax.plot(ssp245_x_mean, ssp245_y_mean, "C4--", label="ssp245")
    ax.plot(ssp585_x_mean, ssp585_y_mean, "C5--", label="ssp585")
    ax.legend()
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.set_title("Future mean steric height (relative to Jan 2000)")
    ax.grid()
    fig.savefig('figures/compare_models_ssp.pdf', dpi = 400)
    plt.show()
    print(y_ssp126[-1])
    print(y_ssp245[-1])
    print(y_ssp585[-1])
    print(y_ssp126[-1]-ssp126_y_mean[-1])
    print(y_ssp245[-1]-ssp245_y_mean[-1])
    print(y_ssp585[-1]-ssp585_y_mean[-1])
    return True

def determine_precentiles(sigmas, ndim):
    mcmc = np.zeros((ndim,7))
    mean = np.zeros(ndim)
    for i in range(ndim):
        mcmc[i,:] = np.percentile(flat_samples[:, :, i], [100-sigmas[2],
                                                     100-sigmas[1],
                                                     100-sigmas[0],
                                                     50,
                                                     sigmas[0],
                                                     sigmas[1],
                                                     sigmas[2],])
        q = np.diff(mcmc[i,2:5])
        txt = "{3} = {0:.6f}_{{-{1:.6f}}}^{{+{2:.6f}}}"
        txt = txt.format(mcmc[i,3], q[0], q[1], labels[i])
        print(txt)
        mean[i] = mcmc[i,3]
    return mcmc, mean

def plot_temps(T1, T2, T3, time):
    fig, ax = plt.subplots(dpi = 400)
    ax.plot(time, T1, label = 'ssp126')
    ax.plot(time, T2, label = 'ssp245')
    ax.plot(time, T3, label = 'ssp585')
    ax.legend()
    ax.set_xlabel("Year")
    ax.set_ylabel("°C (equivalently Kelvin)")
    ax.set_title("Temperature Increase (relative to Jan 2000)")
    ax.grid()
    fig.savefig('figures/temperature_ssp.pdf', dpi = 400)
    plt.show()
    return True

def plot_MSE(y, temp, method):
    length = 100
    a_test   = np.linspace(mcmc[0,3] - 10*mcmc[0,3], mcmc[0,3] + 10*mcmc[0,3], length)
    b_test   = np.linspace(mcmc[1,3] - 20*mcmc[1,3], mcmc[1,3] + 20*mcmc[1,3], length)
    tau_test = np.linspace(mcmc[2,3] - .75*mcmc[2,3], mcmc[2,3] + 10*mcmc[2,3], length)
    S0_test  = np.linspace(mcmc[3,3] - 100*mcmc[3,3], mcmc[3,3] + 100*mcmc[3,3], length)
    mse_a = np.zeros(len(a_test))
    mse_b = np.zeros(len(a_test))
    mse_tau = np.zeros(len(a_test))
    mse_S0 = np.zeros(len(a_test))
    for i in range(length):
        mse_a[i] = MSE(y, 
                     steric_model(np.concatenate((temperature,temp),axis=None),
                                  a_test[i], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(y):])
        mse_b[i] = MSE(y, 
                     steric_model(np.concatenate((temperature,temp),axis=None),
                                  mcmc[0,3], b_test[i], mcmc[2,3], mcmc[3,3])[-len(y):])
        mse_tau[i] = MSE(y, 
                     steric_model(np.concatenate((temperature,temp),axis=None),
                                  mcmc[0,3], mcmc[1,3], tau_test[i], mcmc[3,3])[-len(y):])
        mse_S0[i] = MSE(y, 
                     steric_model(np.concatenate((temperature,temp),axis=None),
                                  mcmc[0,3], mcmc[1,3], mcmc[2,3], S0_test[i])[-len(y):])
    
    fig, axs = plt.subplots(2, 2, dpi = 400, figsize = (10,6), constrained_layout=True)
    fig.suptitle(r'MSE for different parameters $a, b, \tau, S_0$ for tas from {}'.format(method))
    axs[0, 0].plot(a_test, mse_a, label = 'MSE')
    axs[0, 0].axvline(mcmc[0,3], color = 'r', label = r'$a={}$'.format(np.round(mcmc[0,3],decimals=3)))
    axs[0, 0].set_title(r'$a$')
    axs[0, 0].set_ylabel(r'MSE')
    axs[0, 0].grid()
    axs[0, 0].legend(loc = 'upper center')
    axs[0, 1].plot(b_test, mse_b, label = 'MSE')
    axs[0, 1].axvline(mcmc[1,3], color = 'r', label = r'$b={}$'.format(np.round(mcmc[1,3],decimals=3)))
    axs[0, 1].set_title(r'$b$')
    axs[0, 1].set_ylabel(r'MSE')
    axs[0, 1].grid()
    axs[0, 1].legend(loc = 'upper center')
    axs[1, 0].plot(tau_test, mse_tau, label = 'MSE')
    axs[1, 0].axvline(mcmc[2,3], color = 'r', label = r'$\tau={}$'.format(int(mcmc[2,3])))
    axs[1, 0].set_title(r'$\tau$')
    axs[1, 0].set_ylabel(r'MSE')
    axs[1, 0].grid()
    axs[1, 0].legend(loc = 'upper center')
    axs[1, 1].plot(S0_test, mse_S0, label = 'MSE')
    axs[1, 1].axvline(mcmc[3,3], color = 'r', label = r'$S_0={}$'.format(np.round(mcmc[3,3],decimals=4)))
    axs[1, 1].set_title(r'$S_0$')
    axs[1, 1].set_ylabel(r'MSE')
    axs[1, 1].grid()
    axs[1, 1].legend(loc = 'upper center')
    fig.savefig('figures/MSE_{}.pdf'.format(method), dpi = 400)
    plt.show()
    return True

def plot_compare_params(temp_fut, temp_fut2, temp_fut3, x_fut, x_hist):
    fig, ax = plt.subplots(dpi = 400)
    y_ssp585 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp585_1 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,5], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp585_2 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,1], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    ax.plot(x_fut, y_ssp585_1, "r", label=r"$a = {}$".format(np.round(mcmc[0,5], decimals=2)))
    ax.plot(x_fut, y_ssp585_2, "g", label=r"$a = {}$".format(np.round(mcmc[0,1], decimals=2)))
    ax.plot(x_fut, y_ssp585, "C5", label=r"$a = {}$".format(np.round(mcmc[0,3], decimals=2)))
    ax.plot(ssp585_x_mean, ssp585_y_mean, "C5--", label="ssp585")
    ax.legend()
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.set_title("Future mean steric height (relative to Jan 2000)")
    ax.grid()
    fig.savefig('figures/params_a.pdf', dpi = 400)
    plt.show()
    
    fig, ax = plt.subplots(dpi = 400)
    y_ssp585 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp585_1 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,5], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp585_2 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,1], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    ax.plot(x_fut, y_ssp585_1, "r", label=r"$b = {}$".format(np.round(mcmc[1,5], decimals=2)))
    ax.plot(x_fut, y_ssp585_2, "g", label=r"$b = {}$".format(np.round(mcmc[1,1], decimals=2)))
    ax.plot(x_fut, y_ssp585, "C5", label=r"$b = {}$".format(np.round(mcmc[1,3], decimals=2)))
    ax.plot(ssp585_x_mean, ssp585_y_mean, "C5--", label="ssp585")
    ax.legend()
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.set_title("Future mean steric height (relative to Jan 2000)")
    ax.grid()
    fig.savefig('figures/params_b.pdf', dpi = 400)
    plt.show()
    
    fig, ax = plt.subplots(dpi = 400)
    y_ssp585 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,3], mcmc[3,3])[-len(x_fut):]
    y_ssp585_1 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,5], mcmc[3,3])[-len(x_fut):]
    y_ssp585_2 = steric_model(np.concatenate((temperature,temp_fut3),axis=None),
                            mcmc[0,3], mcmc[1,3], mcmc[2,1], mcmc[3,3])[-len(x_fut):]
    ax.plot(x_fut, y_ssp585_1, "r", label=r"$\tau = {}$".format(int(np.round(mcmc[2,5], decimals=0))))
    ax.plot(x_fut, y_ssp585_2, "g", label=r"$\tau = {}$".format(int(np.round(mcmc[2,1], decimals=0))))
    ax.plot(x_fut, y_ssp585, "C5", label=r"$\tau = {}$".format(int(np.round(mcmc[2,3], decimals=2))))
    ax.plot(ssp585_x_mean, ssp585_y_mean, "C5--", label="ssp585")
    ax.legend()
    ax.set_xlabel("year")
    ax.set_ylabel("metres")
    ax.set_title("Future mean steric height (relative to Jan 2000)")
    ax.grid()
    fig.savefig('figures/params_tau.pdf', dpi = 400)
    plt.show()

################################### main ###################################

if __name__ == "__main__":
    """
    reading/normalising data and determining the mean over 12 months
    """
    file = "raw_data/CMIP6 steric SSH/cmip6_CMIP_historical_strh_zostoga_gm.dat"
    x,y,x_mean,y_mean = read_dat_file(file)
    file = "raw_data/cmip6_tas_for_steric_analysis/historical/gm_tas_CMIP_historical_ACCESS-CM2_r1i1p1f1_gn.csv"
    x_mean2, temperature, ref = read_csv_history(file)
    file = "raw_data/cmip6_tas_for_steric_analysis/ssp126/gm_tas_ScenarioMIP_ssp126_ACCESS-CM2_r1i1p1f1_gn.csv"
    x_fut, temp_ssp126 = read_csv_fut(file, ref)
    file = "raw_data/cmip6_tas_for_steric_analysis/ssp245/gm_tas_ScenarioMIP_ssp245_ACCESS-CM2_r1i1p1f1_gn.csv"
    x_fut, temp_ssp245 = read_csv_fut(file, ref)
    file = "raw_data/cmip6_tas_for_steric_analysis/ssp585/gm_tas_ScenarioMIP_ssp585_ACCESS-CM2_r1i1p1f1_gn.csv"
    x_fut, temp_ssp585 = read_csv_fut(file, ref)
    file = "raw_data/CMIP6 steric SSH/cmip6_ScenarioMIP_ssp126_strh_zostoga_gm.dat"
    ssp126_x,ssp126_y,ssp126_x_mean,ssp126_y_mean = read_dat_file(file, historical = False)
    ssp126_y_mean = ssp126_y_mean + y_mean[-1] - ssp126_y_mean[0]
    file = "raw_data/CMIP6 steric SSH/cmip6_ScenarioMIP_ssp245_strh_zostoga_gm.dat"
    ssp245_x,ssp245_y,ssp245_x_mean,ssp245_y_mean = read_dat_file(file, historical = False)
    ssp245_y_mean = ssp245_y_mean + y_mean[-1] - ssp245_y_mean[0]
    file = "raw_data/CMIP6 steric SSH/cmip6_ScenarioMIP_ssp585_strh_zostoga_gm.dat"
    ssp585_x,ssp585_y,ssp585_x_mean,ssp585_y_mean = read_dat_file(file, historical = False)
    ssp585_y_mean = ssp585_y_mean + y_mean[-1] - ssp585_y_mean[0]
    
    """
    plot data, and fits with linear least squares
    """
    
    plot_data_mean(x,y,x_mean,y_mean)
    plot_temps(temp_ssp126, temp_ssp245, temp_ssp585, x_fut)
    
    popt, pcov = curve_fit(lin_func,x_mean,y_mean)
    plot_mean_fit(x_mean, y_mean,lin_func(x_mean, *popt), r'LSE fit to $ax + c$')
    
    popt, pcov = curve_fit(poly_func,x_mean,y_mean)
    plot_mean_fit(x_mean, y_mean,poly_func(x_mean, *popt), r'LSE fit to $ax^3 + bx^2 + cx + d$')
    
    popt, pcov = curve_fit(steric_model,temperature,y_mean)
    print(np.round(popt,decimals=6), np.round(np.sqrt(np.diag(pcov)),decimals=5))
    plot_mean_fit(x_mean, y_mean,steric_model(temperature, *popt), r'LSE fit to steric model')
    
    
    """
    emcee (data has been stored in the file samples_dat_disc.csv and samples_dat.csv)
    """
    ndim = 5
    runs = 5000
    var = 32
    flat_samples = np.loadtxt("samples_dat_disc_{}runs.csv".format(runs), delimiter=",")
    flat_samples = flat_samples.reshape(len(flat_samples)//var,var,ndim)
    samples = np.loadtxt("samples_dat_{}runs.csv".format(runs), delimiter=",").reshape(runs,var,ndim)
    
    labels = ["a", "b", "tau", "S_0", "sigma_SL"]
    
    #plot_samples(ndim, samples, labels)
    plot_samples(ndim, flat_samples, labels)
    
    sigmas = 68.2, 95.4, 99.6
    mcmc, mean = determine_precentiles(sigmas, ndim)
    
    fig = corner.corner(flat_samples, labels=labels, truths = mean)
    plt.show()
    
    #plot_check_emcee(flat_samples, size = 1000) #takes long!!!
    
    plot_future(temp_ssp126, 'ssp126')
    plot_future(temp_ssp245, 'ssp245')
    plot_future(temp_ssp585, 'ssp585')
    
    """
    results: comparison to ssp models and determining the MSE
    """
    
    plot_compare_models(temp_ssp126, temp_ssp245, temp_ssp585, x_fut, x_mean)
    
    
    plot_MSE(ssp126_y_mean, temp_ssp126, 'ssp126')
    plot_MSE(ssp245_y_mean, temp_ssp245, 'ssp245')
    plot_MSE(ssp585_y_mean, temp_ssp585, 'ssp585')
    
    
    plot_compare_params(temp_ssp126, temp_ssp245, temp_ssp585, x_fut, x_mean)
    
    
    
    
    
    
