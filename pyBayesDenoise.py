#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:53:09 2020

@author: artsloan
"""
import numpy as np
import pywt
from scipy.stats import norm
from scipy.special import erfcinv

def e_bayes_denoise(data,wav_name,level,noise_est='level_independent',thresh_rule='median'):
    
    # A python Implementation of the Empirical Bayes Thresholding packages from http://CRAN.R-project.org/package=EbayesThresh.
    
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
         wthr[i] = e_bayes_thresh(lev,vscale,thresh_rule,'decimated')
    
    # Reconstructs the signal from the thresholded levels
    wthr.insert(0,coeffs[0])
    new_data = pywt.waverec(wthr,wav_name)
    if data.size%2 == 1:
        new_data = new_data[:-1]
    sse = np.sum((data-new_data)**2)
    
    extra = {'old_coeffs':coeffs,'new_coeffs':wthr,'SSE':sse}
    return new_data, extra


def e_bayes_thresh(level_coeff,vscale,thresh_rule,trans_type,max_iter=50,min_std=1e-9):
    norm_fac = 1/(-np.sqrt(2)*erfcinv(2*0.75))
    if vscale == 'level_dependent':
        std_est = norm_fac*np.median(np.abs(level_coeff-np.median(level_coeff)))*np.ones_like(level_coeff)
    else:  
        std_est = vscale*np.ones_like(level_coeff)
    
    std_est[std_est<min_std] = min_std
    std_coeff = level_coeff/std_est
    
    weight = weight_from_data(std_coeff,30,trans_type)
    if thresh_rule == 'median':   
        mu_hat, delta = post_med_cauchy(std_coeff,weight,max_iter)
    elif thresh_rule == 'mean':
        mu_hat = post_mean_cauchy(std_coeff,weight)
    elif  np.isin(thresh_rule,['soft','hard']):
        thr = thresh_from_weight(weight, max_iter)
        mu_hat = pywt.threshold(std_coeff,thr,mode=thresh_rule)
        
    return mu_hat*std_est

def thresh_from_weight(weight,max_iter):
    
    zz = np.zeros_like(weight)
    hi_thr = 20;
    thr, delta = interval_solve(zz,cauchy_thresh_zero,0,hi_thr,max_iter,weight=weight)
    
    return thr, delta

def cauchy_thresh_zero(z,w):
    z_norm = norm.cdf(z,0,1)
    d_norm = norm.pdf(z,0,1)
    d1 = np.sqrt(2*np.pi)*d_norm
    y = z_norm - z*d_norm-1/2-(z**2*d1*(1/w-1))/2
    
    return y

def post_med_cauchy(data,weight,max_iter):
    data = data.astype('float')
    mu_hat = np.zeros_like(data)
    weight = weight*np.ones_like(mu_hat)
    mag_data = np.abs(data)
    mag_data_tmp = np.copy(mag_data)
    idx = mag_data < 20
    mag_data[np.invert(idx)] = np.nan
    lo = np.zeros(1)
    
    mu_hat, delta = interval_solve(np.zeros_like(mag_data),
                                   cauchy_med_zero,
                                   lo,
                                   np.nanmax(mag_data),
                                   max_iter,
                                   mag_data=mag_data,
                                   weight=weight)
    

    mu_hat[np.invert(idx)] = mag_data_tmp[np.invert(idx)]-2/mag_data_tmp[np.invert(idx)];
    
    mu_hat[mu_hat < 1e-7] = 0;
    mu_hat = np.sign(data)*mu_hat;
    
    huge_mu_inds = (np.abs(mu_hat) > np.abs(data))
    mu_hat[huge_mu_inds] = data[huge_mu_inds];
    
    return mu_hat, delta
    
def post_mean_cauchy(data,weight):
    exp_data = np.exp(-data**2/2)
    z = weight*(data-(2*(1-exp_data))/data)
    z = z/(weight*(1-exp_data)+(1-weight)*exp_data*data**2)
    mu_hat = z
    
    mu_hat[data==0] = 0
    huge_mu_inds = (np.abs(mu_hat) > np.abs(data))
    mu_hat[huge_mu_inds] = data[huge_mu_inds]
    
    return mu_hat

def cauchy_med_zero(mu_hat,x,weight):
    y = x - mu_hat
    with np.errstate(invalid='ignore'):
        fx = norm.pdf(y,0,1)
        yr = norm.cdf(y,0,1)-x*fx+((x*mu_hat-1)*fx*norm.cdf(-mu_hat,0,1)/norm.pdf(mu_hat,0,1))
    yl = 1+np.exp(-x**2/2)*(x**2*(1/weight-1)-1)
    z = yl/2-yr
    return z


def interval_solve(zf,fun,lo,hi,max_iter,mag_data=[],weight=[]):
    lo = np.asarray(lo*np.ones_like(zf))
    hi = np.asarray(hi*np.ones_like(zf))
    
    tol = 1e-9
    
    num_iter = 0
    con_tol = np.inf
    delta = np.array([])
    
    while con_tol > tol:
        mid_point = (lo+hi)/2
        if (np.asarray(mag_data).size != 0 ) and (np.asarray(weight).size != 0):
            f_mid_point = fun(mid_point,mag_data,weight)
        elif (np.asarray(weight).size != 0 ):
            f_mid_point = fun(mid_point,weight)
        else:
            f_mid_point = fun(mid_point)
        
        with np.errstate(invalid='ignore'):
            idx = f_mid_point <= zf
        
        lo[idx] = mid_point[idx]
        hi[np.invert(idx)] = mid_point[np.invert(idx)]
        delta = np.append(delta,np.max(np.abs(hi-lo)))
        con_tol = np.max(delta[num_iter])
        num_iter += 1
        
        if num_iter > max_iter:
            break
        
    T = (lo+hi)/2
    return T, delta
    
        
def weight_from_data(x,max_iter,trans_type):
    m = len(x)
    
    tmp_weight = np.asarray(np.nan)
    
    if trans_type == 'decimated':
        thr = np.sqrt(2*np.log(m))
    elif trans_type == 'nondecimated':
        thr = np.sqrt(2*np.log(m*np.log2(m)))
    
    wlo = np.asarray(weight_from_thresh(thr))
    
    beta = beta_cauchy(x)
    beta = np.minimum(beta,float(10**20))
    
    whi = np.asarray(np.ones_like(tmp_weight))
    delta_weight = whi-wlo
    
    shi = np.sum(beta/(1+beta))
    
    shi_pos = shi >= 0 
    if np.any(shi_pos):
        tmp_weight[shi_pos] = 1
        if np.all(shi_pos):
            weight = tmp_weight
            return weight
    slo = np.sum(beta/(1+wlo*beta))
    slo_neg = slo<=0
    if np.any(slo_neg):
        tmp_weight[slo_neg] = wlo[slo_neg]
        init_wlo = wlo[slo_neg]
    
    con_tol = np.inf
    w_tol = 100*np.finfo(float).eps
    s_tol = 1e-7
    ii = 0
    
    while con_tol >= w_tol:
        wmid = np.sqrt(whi*wlo)
        smid = np.sum(beta/(1+wmid*beta))
        smid_zero = np.abs(smid) < s_tol
        if np.any(smid_zero):
            tmp_weight[smid_zero] = wmid[smid_zero]
            if not np.any(np.isnan(tmp_weight)):
                weight = tmp_weight
                return weight
        smid_pos = smid > 0
        
        wlo[smid_pos] = wmid[smid_pos]
        smid_neg = smid < 0
        whi[smid_neg] = wmid[smid_neg]
        
        ii += 1
        
        delta_weight = np.append(delta_weight,whi-wlo)
        con_tol = np.abs(np.max(delta_weight[ii])-delta_weight[ii-1])
        
        if ii > max_iter-1:
            break
    
    tmp_weight = np.asarray(np.sqrt(wlo*whi))
    tmp_weight[shi_pos] = 1
    
    if np.any(slo_neg):
        tmp_weight[slo_neg] = init_wlo

    weight = tmp_weight
    return weight
    
    
def beta_cauchy(x):
    phi = norm.pdf(x)
    
    beta = -0.5*np.ones_like(x)
    with np.errstate(divide='ignore'):
        beta[x!=0] = (norm.pdf(0)/phi[x!=0]-1)/x[x!=0]**2-1
    
    return beta
    
def weight_from_thresh(thr):
    fx = norm.pdf(thr,0,1)
    Fx = norm.cdf(thr,0,1)
    weight = np.asarray(1+(Fx - thr*fx-0.5)/(np.sqrt(np.pi/2)*fx*thr**2))
    weight[np.isinf(weight)] = 1
    return 1/weight