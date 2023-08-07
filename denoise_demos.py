#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:20:08 2021

@author: artsloan
"""

import numpy as np
import matplotlib.pyplot as plt
import e_bayes_thresh as ebt
from scipy.signal import savgol_filter
#%% Functions
def gaussian(x,height,center,width):
    return height * np.exp(-((x-center)/(0.60056120439323*width))**2)
rng = np.random.default_rng()
plt.close('all')
#%% User adjustable parameters

x = np.linspace(1,300,1000)
# Peak 1
peak_pos1 = 75
peak_w1 = 3

# Peak 2
peak_pos2 = 126
peak_w2 = 10

# Peak3
peak_pos3 = 200
peak_w3 = 30

wavelet_type = 'sym4'
noise_level = 0.1
sig2noise = 1/noise_level


peak_pos = [peak_pos1, peak_pos2, peak_pos3]
peak_w = [peak_w1, peak_w2, peak_w3]


#%% 

y = (gaussian(x, 1, peak_pos[0], peak_w[0]) +
     gaussian(x, 1, peak_pos[1], peak_w[1]) +
     gaussian(x, 1, peak_pos[2], peak_w[2]))
    
peak_height = list()
resid = list()
snr_improvement = list()
levels = list()
unique_levels = list()

plot_range = np.arange(np.round(0.2*x.size),np.round(0.8*x.size),dtype=int)
for i in range(0,40):
    level = int(1 + np.round(i/4))
    
    
    yn = noise_level*rng.standard_normal(x.size)
    noise_std = np.std(yn)
    snr = 1/noise_std
    
    z = y + yn
    z_denoised,ey = ebt.wavelet_denoise(z, level, wav_name=wavelet_type)
    yn_denoised,en = ebt.wavelet_denoise(yn,level,wav_name=wavelet_type)
    snr_denoised = 1/np.std(yn_denoised)
    
    resid.append(np.std(y[plot_range]-z_denoised[plot_range]))
    peak_height.append(np.max(z_denoised))
    snr_improvement.append(snr_denoised/snr)
    levels.append(level)
    if not level in unique_levels:
        unique_levels.append(level)
        fig, axs = plt.subplots(nrows=2,ncols=1)
        axs[0].set_title('Level: {}'.format(level))
        axs[0].plot(x[plot_range],y[plot_range],zorder=2,label='Original Signal')
        axs[0].plot(x[plot_range],z[plot_range],zorder=1,label='Noisy Signal')
        axs[0].legend()
        
        axs[1].plot(x[plot_range],y[plot_range],zorder=2,label='Original Signal')
        axs[1].plot(x[plot_range],z_denoised[plot_range],zorder=1,label='Denoised Signal')
        axs[1].legend()
        plt.show()
    
plt.figure()
plt.plot(levels,peak_height,'o')
plt.title('Height of denoised peak (true value = 1.0)')
plt.xlabel('wavelet level')
plt.show()

plt.figure()
plt.plot(levels,resid,'o')
plt.title('Closeness to underlying noisless peak')
plt.ylabel('Standard Deviation of Residual')
plt.xlabel('wavelet level')
plt.show()

plt.figure()
plt.plot(levels,np.sqrt(snr_improvement),'o')
plt.title('Signal-to-noise ratio improvement')
plt.ylabel('Square root of signal-to-noise ratio improvement factor')
plt.xlabel('wavelet level')
plt.show()
