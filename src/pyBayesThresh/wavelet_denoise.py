#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:12:45 2023

@author: artsloan
"""

import numpy as np
import pywt
from scipy.special import erfcinv
from . import e_bayes_thresh


def wavelet_denoise(data, level, wav_name='sym4',noise_est='level_independent',thresh_rule='median'):
    """
    Uses e_bayes_thresh to threshold the wavelet coefficients for data and then
    reconsturcts the data from the thresholded coefficients in order to denoise
    the data.

    Parameters
    ----------
    data : ndarray
        The data to be denoised.
    level : int
        The number of levels to use in the reconstruction.
    wav_name : str, optional
        The wavelt to use. Accepts any wavelt supported by pywt. The default is 'sym4'.
    noise_est : str, optional
        Controls the scale used at different levels of the transform. 
        If vscale is a scalar quantity, then it will be assumed that the 
        wavelet coefficients at every level have this standard deviation. 
        If vscale = "level_independent", the standard deviation will be 
        estimated from the highest level of the wavelet transform and will then 
        be used for all levels processed. If vscale="level", then the standard 
        deviation will be estimated separately for each level processed,
        allowing standard deviation that is level-dependent. The default is 
        'level_independent'.
    thresh_rule : str, optional
        Specifies the thresholding rule to be applied to the coefficients. 
        Possible values are median (use the posterior median); mean 
        (use the posterior mean); hard (carry out hard thresholding); 
        soft (carry out soft thresholding). The default is 'median'.

    Returns
    -------
    new_data : ndarray
        The denoised data.
    extra : dict
        A dict containing the old wavelet coefficents, new coefficients, and
        the SSE of the denoising.

    """
    # Decomposes the 1d signal in data into its constituent wavelets, to a specified number of levels with checks to ensure that the maximum useful level is not exceded 
    w = pywt.Wavelet(wav_name)
    max_lev = pywt.dwt_max_level(len(data),w.dec_len)
    
    if level > max_lev:
        level = max_lev
    coeffs = pywt.wavedec(data,wav_name,level=level)
    
    # Order of level coefficients from pywt is [cA_n, cD_n cD_n-1 ... cD_1]
    d1 = coeffs[-1]
    
    if noise_est.lower() == 'level_independent':
        norm_fac = 1/(-np.sqrt(2)*erfcinv(2*0.75))
        vscale = norm_fac*np.median(np.abs(d1))
    elif noise_est.lower() == 'level_dependent':
        vscale = noise_est
        
        
    # Returns thresholded levels
    wthr = coeffs[1:]
    for i , lev in enumerate(wthr):
         wthr[i] = e_bayes_thresh(lev, sdev=vscale, thresh_rule=thresh_rule, trans_type='decimated')
    
    # Reconstructs the signal from the thresholded levels
    wthr.insert(0,coeffs[0])
    new_data = pywt.waverec(wthr,wav_name)
    
    if data.size%2 == 1:
        new_data = new_data[:-1]
    sse = np.sum((data-new_data)**2)
    
    extra = {'old_coeffs':coeffs,'new_coeffs':wthr,'SSE':sse}
    return new_data, extra