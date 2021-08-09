#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:21:27 2021

@author: artsloan
"""
import numpy as np
import pywt
from scipy.stats import norm
from scipy.special import erfcinv

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

# def e_bayes_thresh(level_coeff,vscale,thresh_rule,trans_type,max_iter=50,min_std=1e-9):
    
    
#     norm_fac = 1/(-np.sqrt(2)*erfcinv(2*0.75))
#     if vscale == 'level_dependent':
#         std_est = norm_fac*np.median(np.abs(level_coeff-np.median(level_coeff)))*np.ones_like(level_coeff)
#     else:  
#         std_est = vscale*np.ones_like(level_coeff)
    
#     std_est[std_est<min_std] = min_std
#     std_coeff = level_coeff/std_est
    
#     weight = weight_from_data(std_coeff,30,trans_type)
#     if thresh_rule == 'median':   
#         mu_hat, delta = post_med_cauchy(std_coeff,weight,max_iter)
#     elif thresh_rule == 'mean':
#         mu_hat = post_mean_cauchy(std_coeff,weight)
#     elif  np.isin(thresh_rule,['soft','hard']):
#         thr = thresh_from_weight(weight, max_iter)
#         mu_hat = pywt.threshold(std_coeff,thr,mode=thresh_rule)
        
#     return mu_hat*std_est


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

def post_mean(x,s=1,w=0.5,prior='cauchy',a=0.5):
    """
    Given a single value or a vector of data and sampling standard deviations 
    (sd equals 1 for Cauchy prior), find the corresponding posterior mean 
    estimate(s) of the underlying signal value(s)

    Parameters
    ----------
    x : TYPE
        A data value or a vector of data.
    s : TYPE, optional
        A single value or a vector of standard deviations if the Laplace prior is used. If
        a vector, must have the same length as x. Ignored if Cauchy prior is used. The default is 1.
    w : TYPE, optional
        The value of the prior probability that the signal is nonzero.. The default is 0.5.
    prior : string, optional
        Family of the nonzero part of the prior; can be "cauchy" or "laplace".. The default is 'cauchy'.
    a : float, optional
        The scale parameter of the nonzero part of the prior if the Laplace prior is used. The default is 0.5.

    Returns
    -------
    mu_hat : ndarray
        If x is a scalar, the posterior mean E(θ|x) where θ is the mean of the
        distribution from which x is drawn. If x is a vector with elements 
        x1,...,xn and s is a vector with elements s1,...,sn 
        (s_i is 1 for Cauchy prior), then the vector returned has elements 
        E(θi|xi,si), where each xi has mean θiand standard deviation si, 
        all with the given prior.

    """
    
    if prior == 'laplace':
        mu_hat = post_mean_laplace(x,s,w,a=a)
    elif prior == 'cauchy':
        mu_hat = post_med_cauchy(x,w)
    
    return mu_hat

        
def post_mean_laplace(x,s,w,a):
    a = np.min([a,20])
    
    wpost = wpost_laplace(w, x, s, a)
    
    sx = np.sign(x)
    x = np.abs(x)
    xpa = x/s + s*a
    xma = x/s - s*a
    xpa[xpa >35] = 35
    xma[xma<-35] = -35
    cp1 = norm.cdf(xma,0,1)
    cp2 = norm.cdf(-xpa,0,1)
    ef = np.exp(np.minimum(2*a*x,100))
    post_mean_cond = x - a * s**2 *(2*cp1/(cp1+ef+cp2)-1)
    
    return (sx * wpost * post_mean_cond)

def wpost_laplace(w,x,s=1,a=0.5):
    #  Calculate the posterior weight for non-zero effect
    return 1-(1-w)/(1+w*beta_laplace(x,s,a))


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
    """
    The objective function that has to be zeroed, component by component,
    to find the posterior median when the quasi-Cauchy prior is used.  
    
    x and z may be scalars.

    Parameters
    ----------
    mu_hat : ndarray
        parameter vector.
    x : ndarray
        data vector.
    weight : ndarray
        weight.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y = x - mu_hat
    fx = norm.pdf(y,0,1)
    
    yleft = norm.cdf(y,0,1)-x*fx+((x*mu_hat-1)*fx*norm.cdf(-mu_hat,0,1)/norm.pdf(mu_hat,0,1))
    yright = 1+np.exp(-x**2/2)*(x**2*(1/weight-1)-1)
    
    return yright/2 - yleft

def cauchy_thresh_zero(z,w):
    """
    The objective function that has to be zeroed to find the Cauchy threshold. 

    Parameters
    ----------
    z : ndarray
        putative threshold vector.
    w : ndarray
        weigh.

    Returns
    -------
    y : ndarray
        objective function value

    """
    z_norm = norm.cdf(z,0,1)
    d_norm = norm.pdf(z,0,1)
    y = z_norm - z*d_norm-1/2-(z**2*np.exp(-z**2/2)*(1/w-1))/2 # From R version of the code
    # d1 = np.sqrt(2*np.pi)*d_norm # From the MATLAB version of the code 
    # y = z_norm - z*d_norm-1/2-(z**2*d1*(1/w-1))/2
    
    return y

def laplace_thresh_zero(x,s=1,w=0.5,a=0.5):
    """
    The function that has to be zeroed to find the threshold with the
    Laplace prior.  Only allow a < 20 for input value.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    s : TYPE, optional
        DESCRIPTION. The default is 1.
    w : TYPE, optional
        DESCRIPTION. The default is 0.5.
    a : TYPE, optional
        DESCRIPTION. The default is 0.5.

    Returns
    -------
    z : TYPE
        DESCRIPTION.

    """
    a = np.min(a,20)
    xma = x/s - s*a
    z = norm.cdf(xma) -1/a * (1/s*norm.pdf(xma)) * (1/w + beta_laplace(x,s,a))
    
    return z

def interval_solve(zf,fun,lo,hi,max_iter=50,**kwargs):
    """
    Python implementation of the function vecbinsolv from the R package
    
    Given a monotone function fun, and a vector of values
    zf find a vector of numbers t such that f(t) = zf.
    The solution is constrained to lie on the interval (lo, thi)

    The function fun may be a vector of increasing functions 
    
    It is important that fun should work for vector arguments.

    Works by successive bisection, carrying out max_iter harmonic bisections
    of the interval between lo and hi, or until the error beteween the fuction 
    value and the target value falls below 1e-9

    Parameters
    ----------
    zf : ndarray
        Target value(s) of the function, may be a scalar or vector.
    fun : function
        The target fuction for the solution.
    lo : float
        Lower limit of the function value.
    hi : float
        Upper limit of the function value.
    max_iter : int, optional
        Maximum number of allowed iterations. The default is 50.
    mag_data : TYPE, optional
        DESCRIPTION. The default is [].
    weight : TYPE, optional
        DESCRIPTION. The default is [].

    Returns
    -------
    T : TYPE
        DESCRIPTION.
    delta : TYPE
        DESCRIPTION.

    """
    
    lo = np.asarray(lo*np.ones_like(zf),dtype=float)
    hi = np.asarray(hi*np.ones_like(zf),dtype=float)
    
    tol = 1e-9
    
    num_iter = 0
    con_tol = np.inf
    delta = np.array([])
    
    while con_tol > tol:
        mid_point = (lo+hi)/2
        f_mid_point = fun(mid_point,**kwargs)
        idx = f_mid_point <= zf
        lo[idx] = mid_point[idx]
        hi[~idx] = mid_point[~idx]
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
    '''
    Given a value or vector x of values, find the value(s) of the function 
    β(x) = g(x)/φ(x) −1, where g is the convolution of the quasi-Cauchy with 
    the normal density φ(x).
    
    g(x) = \dfrac{1}{\sqrt{2\pi}} x^{-2}(1-e^{-x^2/2})
    
    

    Parameters
    ----------
    x : ndarray
        A vector of real values.

    Returns
    -------
    beta : ndarray
        A vector the same length as x, containing the value(s) β(x)

    '''
    phix = norm.pdf(x)
    j = x != 0
    beta = x
    
    beta[~j] = -1/2
    
    beta[j] = (norm.pdf(0)/phix[j] - 1)/x[j]**2-1
    
    return beta

def beta_laplace(x, s=1, a=0.5):
    """
    Given a single value or a vector of x and s, find the value(s) of the 
    function β(x; s,a) = g(x; s,a)/fn(x; 0,s)−1, where fn(x; 0,s) is the 
    normal density with mean 0 and standard deviation s, and g is the 
    convolution of the Laplace density with scale parameter a, γa(μ), with the
    normal density fn(x; μ,s) with mean mu and standard deviation s.
    
    The Laplace density is given by γ(u; a) = 1/2 ae**(−a|u|) and is also 
    known as the double exponentia density.

    Parameters
    ----------
    x : ndarray
        the value or vector of data values.
    s : ndarray, optional
        The value or vector of standard deviations; if vector, must have the same length
        as x. The default is 1.
    a : float, optional
        The scale parameter of the Laplace distribution. The default is 0.5.

    Returns
    -------
    beta : ndarray
        A vector of the same length as x is returned, containing the value(s) beta(x).

    """
    x = np.abs(x)
    xpa = x/s +s*a
    xma = x/s -s*a
    rat1 = 1/xpa
    rat1[xpa > 35] = norm.cdf(-xpa[xpa <35],0,1)/norm.pdf(xpa[xpa < 35],0,1)
    rat2 = 1/np.abs(xma)
    xma[xma > 35] = 35
    rat2[xma > -35] = norm.cdf(xma[xma > -35],0,1)/norm.pdf(xma[xma > -35],0,1)
    beta = (a * s) / 2 * (rat1 + rat2) - 1
    
    return beta
    
    
def weight_from_thresh(thr,s=1,prior='cauchy',a=0.5):
    
    """
    Given a value or vector of thresholds and sampling standard deviations (sd equals 1 for Cauchy
    prior), find the mixing weight for which this is(these are) the threshold(s) of the posterior median
    estimator. If a vector of threshold values is provided, the vector of corresponding weights is returned.

    Parameters
    ----------
    thr : ndarray
        Threshold value or vector of values.
    s : ndarray
        A single value or a vector of standard deviations if the Laplace prior is used. If
        a vector, must have the same length as thr. Ignored if Cauchy prior is used.
    prior : string
        Specification of prior to be used; can be "cauchy" or "laplace".
    a : float
        Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.

    Returns
    -------
    ndarray
        The numerical value or vector of values of the corresponding weight is returned.

    """
    if prior == 'laplace':
        tma = thr/s - s*a
        weight = 1/np.abs(tma)
        j = tma > -35
        
        fx = norm.pdf(tma[j],0,1)
        Fx = norm.cdf(tma[j],0,1)
        
        weight[j] = Fx/fx
        weight = a*s*weight - beta_laplace(thr, s, a)
        
    elif prior == 'cauchy':
        fx = norm.pdf(thr,0,1)
        Fx = norm.cdf(thr,0,1)
        weight = np.asarray(1+(Fx - thr*fx-0.5)/(np.sqrt(np.pi/2)*fx*thr**2))
        weight[np.isinf(weight)] = 1
    return 1/weight

def thresh_from_weight(w,s=1,prior='cauchy', bayesfac=False, a=0.5, max_iter=50):
    if bayesfac:
        z = 1/w - 2
        if prior == 'cauchy':
            zz = z*np.ones_like(z)
            thr, delta = interval_solve(z, beta_cauchy, 0, 20)
            
        elif prior == 'laplace':
            zz = z*np(np.ones_like(s))
            thr, delta = interval_solve(zz,beta_laplace,0,10,s=s,a=a)
    else:
        zz = np.zeros_like(w)
        hi_thr = 20;
        if prior == 'cauchy':
            thr, delta = interval_solve(zz,cauchy_thresh_zero,0,hi_thr,max_iter,w=w)
        elif prior == 'laplace':
            thr, delta = interval_solve(zz,laplace_thresh_zero,0,s*(25+s*a),max_iter,w=w)
    return thr, delta