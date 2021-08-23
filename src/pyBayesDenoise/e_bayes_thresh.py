#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 14:21:27 2021

@author: artsloan
"""
#%% Import Statements
import numpy as np
import pywt
from scipy.stats import norm
from scipy.special import erfcinv
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

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


def e_bayes_thresh(x, 
                   prior='cauchy', 
                   a=0.5, 
                   bayesfac=False, 
                   sdev=None, 
                   vebose=False, 
                   thresh_rule='median', 
                   universal_thresh=True, 
                   trans_type='decimated', 
                   max_iter=50, 
                   min_std=1e-9):
    """
    Given a sequence of data, performs Empirical Bayes thresholding, as discussed in Johnstone and
    Silverman (2004).

    Parameters
    ----------
    x : ndarray
        Vector of data values.
    prior : str, optional
        Specification of prior to be used conditional on the mean being 
        nonzero; can be "cauchy" or "laplace". The default is 'cauchy'.
    a : float, optional
        Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
        If, on entry, a = NA and prior = "laplace", then the scale parameter 
        will also be estimated by marginal maximum likelihood. The default is 0.5.
    bayesfac : bool, optional
        If bayesfac = TRUE, then whenever a threshold is explicitly calculated,
        the Bayes factor threshold will be used. The default is False.
    sdev : TYPE, optional
        DESCRIPTION. The default is None.
    thresh_rule : str, optional
        Specifies the thresholding rule to be applied to the data. Possible 
        values are "median" (use the posterior median); "mean" 
        (use the posterior mean); "hard" (carry out hard thresholding);
        "soft" (carry out soft thresholding). The default is 'median'.
    universal_thresh : bool, optional
        If universalthresh = TRUE, the thresholds will be upper bounded by
        universal threshold; otherwise, the thresholds can take any
        non-negative values. The default is True.
    trans_type : TYPE, optional
        DESCRIPTION. The default is 'decimated'.
    max_iter : int, optional
        Maximum number of solver iterations. The default is 50.
    min_std : float, optional
        Minimum standard deviation to prevent divide by zero issues.
        The default is 1e-9.

    Returns
    -------
    ndarray
        The estimated mean vector

    """
    if sdev is None:
        norm_fac = 1/(-np.sqrt(2)*erfcinv(2*0.75))
        std_est = norm_fac*np.median(np.abs(x-np.median(x)))*np.ones_like(x)
        stabadjustment = True
        
    elif np.asarray(sdev).size == 1:
        std_est = sdev*np.ones_like(x)
        stabadjustment = True
    else:
        std_est = sdev
            
    std_est[std_est<min_std] = min_std
    if stabadjustment:
        sdev = std_est
        std_est = np.mean(std_est)
        x = x/std_est
        s = sdev/std_est
    else:
        s = sdev
    
    if prior == 'laplace' and np.isnan(a):
        w, a = weight_and_scale_from_data(x,s,universal_thresh)
    else:
        w = weight_from_data(x,s,prior,a,universal_thresh,trans_type)
        
    if thresh_rule == 'median':
        mu_hat = post_med(x,s=s,w=w,prior=prior,a=a)
    elif thresh_rule == 'mean':
        mu_hat, delta = post_mean(x,s=s,w=w,prior=prior,a=a)
    elif np.isin(thresh_rule,['soft','hard']):
        thr = thresh_from_weight(w, s=s, prior=prior, bayesfac=bayesfac, a=a, max_iter=max_iter)
        mu_hat = pywt.threshold(x,thr,mode=thresh_rule)
    else:
        mu_hat = np.nan
            
    if stabadjustment:
        return mu_hat*std_est
    else:
        return mu_hat


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
    if prior == 'cauchy':
        mu_hat = post_mean_cauchy(x,w)
    elif prior == 'laplace':
        mu_hat = post_mean_laplace(x,s,w,a=a)
        
    return mu_hat


def post_med(x,s=1,w=0.5,prior='cauchy',a=0.5):
    """
    Given a single value or a vector of data and sampling standard deviations 
    (sd is 1 for Cauchy prior), find the corresponding posterior median
    estimate(s) of the underlying signal value(s).
    
    The routine calls the relevant one of the routines post_med_laplace or
    post_med_cauchy. In the Laplace case, the posterior median is found 
    explicitly, without any need for the numerical solution of an equation. 
    In the quasi-Cauchy case, the posterior median is found by finding the 
    zero, component by component, of the vector function cauchy_med_zero.

    Parameters
    ----------
    x : ndarray
        A data value or a vector of data.
    s : ndarray, optional
        A single value or a vector of standard deviations if the Laplace prior 
        is used. If a vector, must have the same length as x. Ignored if Cauchy
        prior is used. The default is 1.
    w : float, optional
        The value of the prior probability that the signal is nonzero.
        The default is 0.5.
    prior : bool, optional
        Family of the nonzero part of the prior; can be "cauchy" or "laplace". The default is 'cauchy'.
    a : float, optional
        The scale parameter of the nonzero part of the prior if the Laplace 
        prior is used. The default is 0.5.

    Returns
    -------
    mu_hat : ndarray
        The posterior median.

    """
    if prior == 'cauchy':
        mu_hat, delta = post_med_cauchy(x,w)
    elif prior == 'laplace':
        mu_hat = post_med_laplace(x,s,w,a)
        
    return mu_hat


#%% Calculate Weights
def weight_and_scale_from_data(x, s=1, universal_thresh=True):
    """
    Given a vector of data and a single value or vector of sampling standard 
    deviations, find the marginal maximum likelihood choice of both weight and
    scale factor under the Laplace prior.

    Parameters
    ----------
    x : TYPE
        A vector of data.
    s : float, ndarray, optional
        A single value or a vector of standard deviations. If vector, must 
        have the same length as x. The default is 1.
    universal_thresh : TYPE, optional
        If universalthresh = TRUE, the thresholds will be upper bounded by
        universal threshold; otherwise, the thresholds can take any 
        non-negative values. The default is True.

    Returns
    -------
    w : ndarray 
        The estimated weight.
    a : ndarray
        The estimated scale factor.

    """
    
    if universal_thresh:
        thi = np.array(s * np.sqrt(2 * np.log(len(x))))
    else:
        thi = np.inf
        
    tlo = np.zeros_like(s)
    lo = np.array([0,0.04])
    hi = np.array([1,3])
    start_par = np.array([0.5,0.5])
    
    uu  = minimize(laplace_neg_log_likelyhood,start_par, args=(x,s,tlo,thi),method='L-BFGS-B',bounds=((lo[0],hi[0]),(lo[1],hi[1])))
    uu = uu.x
    
    a = uu[1]
    wlo = weight_from_thresh(thi, s=s, prior='laplace', a=a)
    whi = weight_from_thresh(tlo, s=s, prior='laplace', a=a)
    wlo = np.max(wlo)
    whi = np.max(whi)
    w = uu[0] * (whi - wlo) + wlo
    return w, a
 
    
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
    if prior == 'cauchy':
        thr = np.asarray(thr)
        fx = norm.pdf(thr,0,1)
        Fx = norm.cdf(thr,0,1)
        weight = np.asarray( 1 + (Fx - thr * fx - 0.5 ) / (np.sqrt(np.pi / 2) * fx * thr ** 2))
        weight[np.isinf(weight)] = 1
        
    elif prior == 'laplace':
        tma = thr / s - s * a
        weight = np.asarray(1 / np.abs(tma))
        j = tma > -35
        
        fx = norm.pdf(tma[j],0,1)
        Fx = norm.cdf(tma[j],0,1)
        weight[j] = Fx/fx
        
        weight = a * s * weight - beta_laplace(thr, s=s, a=a)
        
    return 1/weight


def weight_from_data(x, s=1, prior='cauchy', a=0.5, universal_thresh=True, 
                     trans_type='decimated', max_iter=50):
    """
    Suppose the vector (x1,...,xn) is such that xi is drawn independently from 
    a normal distribution with mean θi and standard deviation si 
    (s_i equals 1 for Cauchy prior). The prior distribution of the θi is a
    mixture with probability 1 −w of zero and probability w of a given
    symmetric heavy-tailed distribution. This routine finds the marginal 
    maximum likelihood estimate of the parameter w.

    Parameters
    ----------
    x : ndarray
        Vector of data.
    s : float or ndarray, optional
        A single value or a vector of standard deviations if the Laplace prior 
        is used. If a vector, must have the same length as x. Ignored if Cauchy
        prior is used. The default is 1.
    prior : bool, optional
        Specification of prior to be used; can be "cauchy" or "laplace".
        The default is 'cauchy'.
    a : TYPE, optional
        Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
        The default is 0.5.
    universal_thresh : TYPE, optional
        If universalthresh = TRUE, the thresholds will be upper bounded by
        universal threshold; otherwise, the thresholds can take any
        non-negative values. The default is True.
    trans_type : TYPE, optional
        DESCRIPTION. The default is 'decimated'.
    max_iter : TYPE, optional
        Maximum number of solver iterations. The default is 50.

    Returns
    -------
    weight : ndarray
        The numerical value of the estimated weight.

    """
    m = len(x)
    
    if universal_thresh:
        
        if trans_type == 'decimated':
            thr = s*np.sqrt(2*np.log(m))
            
        if trans_type == 'nondecimated':
            thr = s*np.sqrt(2*np.log(m*np.log2(m)))
        
        wlo = np.asarray(weight_from_thresh(thr,s=s,prior=prior,a=a),dtype=np.float64)
        wlo = np.max(wlo)
        
    else:
        wlo = 0
            
    tmp_weight = np.asarray(np.nan)  
      
    if prior == 'cauchy':
        beta = beta_cauchy(x)
    elif prior == 'laplace':
        beta = beta_laplace(x,s=s,a=a)
    
    beta = np.minimum(beta,float(1e20))
    whi = np.asarray(1.0)
    
    delta_weight = whi - wlo
    
    shi = np.sum(beta / (1 + beta))
    shi_pos = shi >= 0
    
    if np.any(shi_pos):
        tmp_weight[shi_pos] = 1
        if np.all(shi_pos):
            weight = tmp_weight
            return weight
        
    slo = np.sum(beta / (1 + wlo * beta))
    slo_neg = slo <= 0
    
    if np.any(slo_neg):
        tmp_weight[slo_neg] = wlo[slo_neg]
        init_wlo = wlo[slo_neg]
    
    con_tol = np.inf
    w_tol = 100*np.finfo(np.float64).eps
    s_tol = 1e-7
    ii = 0
    
    wlo = np.asarray(wlo)
    whi = np.asarray(whi)
    while con_tol > w_tol:
        wmid = np.sqrt(whi*wlo)
        smid = np.sum(beta / (1 + wmid * beta))
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
    
def weight_mono_from_data(x, prior='cauchy',a=0.5,tol=01e-8,max_iter=50):
    """
    Given a vector of data, find the marginal maximum likelihood choice of
    weight sequence subject tothe constraints that the weights are monotone
    decreasing.
    
    Parameters
    ----------
    x : TYPE
        A vector of data.
    prior : bool, optional
        Specification of the prior to be used; can be 'cauchy' or 'laplace'. The default is 'cauchy'.
    a : float, optional
        Scale parameter in prior if prior="laplace". Ignored if prior="cauchy". The default is 0.5.
    tol : TYPE, optional
        Absolute tolerance to within which estimates are calculated.. The default is 01e-8.
    max_iter : TYPE, optional
        Maximum number of solver iterations. The default is 50.

    Returns
    -------
    w : ndarray
        The vector of estimated weights.

    """
    wmin = weight_from_thresh(np.sqrt(2 * np.log(len(x))), prior=prior,a=a)
    winit = 1
    
    if prior == 'cauchy':
        beta = beta_cauchy(x)
    elif prior == 'laplace':
        beta = beta_laplace(x,a=a)
        
    w = winit*np.ones_like(beta)
    
    ii = 0
    
    while ii <= max_iter:
        aa = w + 1 / beta
        ps = w + aa
        ww = 1 / aa ** 2
        iso_reg = IsotonicRegression(increasing=False).fit(ps,ww)
        wnew = iso_reg.predict(ps)
        wnew = np.maximum(wmin,wnew)
        wnew = np.minimum(1,wnew)
        zinc = np.max(np.abs(np.ptp(wnew - w)))
        w = wnew
        if zinc < tol:
            return w
        ii += 1
    return w
        
    
    
        
    
    
#%% Calculate Thresholds
def thresh_from_weight(w, s=1, prior='cauchy', bayesfac=False, a=0.5, max_iter=50):
    """
    Given the vector of weights w and s (sd), find the threshold or
    vector of thresholds corresponding to these weights, under the
    specified prior.
    If bayesfac=TRUE the Bayes factor thresholds are found, otherwise
    the posterior median thresholds are found.
    If the Laplace prior is used, a gives the value of the inverse scale
    (i.e., rate) parameter

    Parameters
    ----------
    w : ndarray
        vector of weights.
    s : float, ndarray, optional
        scalar or vector of standard deviaitons. The default is 1.
    prior : TYPE, optional
        The type of prior to use, allowable values are 'cauchy' and 'laplace'.
        The default is 'cauchy'.
    bayesfac : bool, optional
        Whether to find the Bayes factor threshold or the posterior median.
        The default is False.
    a : float, optional
        Scale parameter. The default is 0.5.
    max_iter : TYPE, optional
        Maximum number of solver iterations. The default is 50.

    Returns
    -------
    thr : ndarray
        Vector of thresholds.
    delta : ndarray
        Vector of solver error for each iteration.

    """
    if bayesfac:
        z = 1 / w - 2
        
        if prior == 'cauchy':
            zz = z * np.ones_like(z)
            thr, delta = interval_solve(z, beta_cauchy, 0, 20)
            
        elif prior == 'laplace':
            zz = z * np(np.ones_like(s))
            thr, delta = interval_solve(zz, beta_laplace, 0, 10, s=s, a=a)
            
    else:
        
        zz = np.zeros(np.max([np.size(w),np.size(s)]))
        hi_thr = 20;
        
        if prior == 'cauchy':
            thr, delta = interval_solve(zz, 
                                        cauchy_thresh_zero, 
                                        0, 
                                        hi_thr, 
                                        max_iter=max_iter, 
                                        w=w)
            
        elif prior == 'laplace':
            thr, delta = interval_solve(zz, 
                                        laplace_thresh_zero, 
                                        0, 
                                        s * (25 + s * a), 
                                        max_iter=max_iter,
                                        s=s,
                                        w=w,
                                        a=a)
            
    return thr, delta

def thresh_from_data(x, s=1, prior='cauchy', bayesfac=False, a=0.5, universal_thresh=True):
    """
    Given the data x, the prior, and any other parameters, find the
    threshold corresponding to the marginal maximum likelihood
    estimator of the mixing weight.
    
    This fucntion just passes things on to the appropriate calculating function
    for the given inputs
    

    Parameters
    ----------
    x : ndarray
        Data Vector.
    s : float or ndarray, optional
        The standard deviation, may be  a vector. The default is 1.
    prior : str, optional
        The type of prior to use, allowable values are 'cauchy' and 'laplace'.
        The default is 'cauchy'.
    bayesfac : bool, optional
        Whether to find the Bayes factor threshold or the posterior median.
        The default is False.
    a : float, optional
        Scale parameter. The default is 0.5.
    universal_thresh : bool, optional
        Whether to use the universal threshold for setting the upper bounds of
        the threshold. The default is True.

    Returns
    -------
    thr : float
        The Threshold values.

    """
    if prior == 'cauchy':
        s = 1
        
    if prior == 'laplace' and np.isnan(a):
        w, a = weight_and_scale_from_data(x,s=s,universal_thresh=universal_thresh)
    else:
        w = weight_from_data(x,s=s,prior=prior,a=a)
        
    thr, delta = thresh_from_weight(w, s=s, prior=prior, bayesfac=bayesfac, a=a)
    
    return thr


#%% Cauchy Prior
def beta_cauchy(x):
    """
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

    """
    x = x.astype(np.float64)
    phix = norm.pdf(x)
    j = x != 0
    beta = x
    
    beta[~j] = -1/2
    
    beta[j] = (norm.pdf(0)/phix[j] - 1)/x[j]**2-1
    
    return beta


def post_mean_cauchy(x, w):
    """
    Find the posterior mean for the quasi-Cauchy prior with mixing
    weight w given data x, which may be a scalar or a vector.

    Parameters
    ----------
    x : ndarray
        Data vector.
    w : float
        Weight.

    Returns
    -------
    mu_hat : ndarray
        The posterior mean.

    """
    exp_x = np.exp(-x ** 2 / 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        z = w * (x - (2 * (1 - exp_x)) / x)
        z = z / (w * (1 - exp_x) + (1 - w) * exp_x * x ** 2)
    mu_hat = z
    mu_hat[x==0] = 0
    huge_mu_inds = (np.abs(mu_hat) > np.abs(x))
    mu_hat[huge_mu_inds] = x[huge_mu_inds]
    
    return mu_hat


def post_med_cauchy(x,w,max_iter=50):
    """
    Find the posterior median of the Cauchy prior with mixing weight w,
    pointwise for each of the data points x

    Parameters
    ----------
    x : ndarray
        Data vector.
    w : float
        Weight.
    max_iter : int, optional
        Maximum number of iterations for the solver. The default is 50.
    
    Returns
    -------
    mu_hat : ndarray
        Poserior median.
    delta : ndarray
        Vector of the solver error for each each iteration..

    """
    x = x.astype(np.float64)
    mu_hat = np.zeros_like(x)
    w = w*np.ones_like(mu_hat)
    mag_x = np.abs(x)
    mag_x_tmp = np.copy(mag_x)
    idx = mag_x < 20
    mag_x[np.invert(idx)] = np.nan
    lo = np.zeros(1)
    
    mu_hat, delta = interval_solve(np.zeros_like(mag_x),
                                   cauchy_med_zero,
                                   lo,
                                   np.nanmax(mag_x),
                                   max_iter,
                                   x=mag_x,
                                   w=w)
    

    mu_hat[np.invert(idx)] = mag_x_tmp[np.invert(idx)]-2/mag_x_tmp[np.invert(idx)];
    
    mu_hat[mu_hat < 1e-7] = 0.0;
    mu_hat = np.sign(x)*mu_hat;
    
    huge_mu_inds = (np.abs(mu_hat) > np.abs(x))
    mu_hat[huge_mu_inds] = x[huge_mu_inds];
    
    return mu_hat, delta


def cauchy_med_zero(mu_hat, x ,w):
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
    ndarray
        objective function value.

    """
    y = x - mu_hat
    fx = norm.pdf(y,0,1)
    
    yleft = norm.cdf(y,0,1) - x * fx + ((x * mu_hat - 1) * fx * norm.cdf(-mu_hat,0,1) / norm.pdf(mu_hat,0,1))
    yright = 1 + np.exp(-x ** 2 / 2) * (x ** 2 * (1 / w - 1) - 1)
    
    return yright / 2 - yleft


def cauchy_thresh_zero(z, w):
    """
    The objective function that has to be zeroed to find the Cauchy threshold. 

    Parameters
    ----------
    z : ndarray
        Putative threshold vector.
    w : ndarray
        Weight.

    Returns
    -------
    y : ndarray
        objective function value

    """
    z_norm = norm.cdf(z,0,1)
    d_norm = norm.pdf(z,0,1)
    y = z_norm - z * d_norm - 1/2 - (z ** 2 * np.exp(-z ** 2 / 2) * (1 / w - 1)) / 2 # From R version of the code

    return y


#%% Laplace Prior
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
    xpa = np.asarray(x/s + s * a)
    xma = np.asarray(x/s - s * a)
    rat1 = np.asarray(1/xpa)
    rat1[xpa < 35] = norm.cdf(-xpa[xpa < 35],0,1)/norm.pdf(xpa[xpa < 35],0,1)
    
    rat2 = np.asarray(1/np.abs(xma))
    xma[xma > 35] = 35
    rat2[xma > -35] = norm.cdf(xma[xma > -35],0,1)/norm.pdf(xma[xma > -35],0,1)
    beta = (a * s) / 2 * (rat1 + rat2) - 1
    
    return beta


def post_mean_laplace(x,s=1,w=0.5,a=0.5):
    """
    Find the posterior mean for the double exponential prior

    Parameters
    ----------
    x : ndarray
        Vector of data/observations.
    s : float or ndarray, optional
        Standard deviation(s), may be a vector. The default is 1.
    w : float, optional
        Weight. The default is 0.5.
    a : float, optional
        Scale parameter. The default is 0.5.

    Returns
    -------
    mu_hat : float
        The poserior mean.

    """
    a = np.min([a,20])
    sx = np.sign(x)
    
    wpost = wpost_laplace(w, x, s, a)
    
    x = np.abs(x)
    xpa = x / s + s * a
    xma = x / s - s * a
    xpa[xpa > 35] = 35
    xma[xma < -35] = -35
    cp1 = norm.cdf(xma,0,1)
    cp2 = norm.cdf(-xpa,0,1)
    ef = np.exp(np.minimum(2 * a * x, 100))
    post_mean_cond = x - a * s**2 * (2 * cp1 / (cp1 + ef * cp2) - 1)
    mu_hat = sx * wpost * post_mean_cond
    
    return mu_hat


def post_med_laplace(x, s=1, w=0.5, a=0.5):
    """
    Find the posterior median for the Laplace prior.

    Parameters
    ----------
    x : ndarray
        Vector of data/observations.
    s : float or ndarray, optional
        Standard deviation(s), may be a vector. The default is 1.
    w : float, optional
        Weight. The default is 0.5.
    a : float, optional
        Scale parameter. The default is 0.5.

    Returns
    -------
    mu_hat : float
        The posterior median.

    """
    # Only allow a < 20 for input value
    a = np.min([a,20])
    
    # Work with the absolute value of x, and for x > 25 use the approximation
    # to norm.pdf(x-a)*beta_laplace(x, a)
    
    sx = np.sign(x)
    x = np.abs(x)
    xma = x / s - s * a
    zz = 1 / a * (1 / s * norm.pdf(xma,0,1)) * (1 / w + beta_laplace(x, s, a))
    zz[xma > 25] = 0.5
    mu_cor = norm.ppf(np.minimum(zz,1),0,1)
    mu_hat = sx * np.maximum(0, xma - mu_cor) * s
    
    return mu_hat 
    

def wpost_laplace(w,x,s=1,a=0.5):
    """
    Calculate the posterior weight for non-zero effect
    
    Parameters
    ----------
    w : float
        weight.
    x : ndarray
        Data vector.
    s : float or ndarray, optional
        Standard deviation. May be a scalar or a vector. The default is 1.
    a : float, optional
        Scale parameter. The default is 0.5.

    Returns
    -------
    float
        The posterior weight.

    """
    return 1-(1-w)/(1+w*beta_laplace(x,s,a))


def laplace_thresh_zero(x,s=1,w=0.5,a=0.5):
    """
    The function that has to be zeroed to find the threshold with the
    Laplace prior.  Only allow a < 20 for input value.

    Parameters
    ----------
    x : ndarray
        Data vector.
    s : float or ndarray, optional
        Standard deviation of the data. May be a sclar or a vector of the same
        lenght as x. The default is 1.
    w : float, optional
        The weight parameter. The default is 0.5.
    a : float, optional
        The scale parameter. The default is 0.5.

    Returns
    -------
    z : ndarray
       Objective function value(s)
        
    """
    
    a = np.min([a,20])
    
    xma = x / s - s * a
    
    z = norm.cdf(xma,0,1) - 1 / a * (1 / s * norm.pdf(xma,0,1)) * (1 / w + beta_laplace(x,s,a))
    
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
    T : float
        The return value of the solver.
    delta : ndarray
        Vector of the solver error for each each iteration.

    """
    
    lo = np.asarray(lo*np.ones_like(zf),dtype=np.float64)
    hi = np.asarray(hi*np.ones_like(zf),dtype=np.float64)
    
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


def laplace_neg_log_likelyhood(xpar, *args):
    
    """
    Marginal negative log likelihood function for laplace prior. Constraints 
    for thresholds need to be passed through *args.

    Parameters
    ----------
    xpar : ndarray
        A two element vector:
            xpar[1] : a value between [0, 1] which will be adjusted to range of w 
            xpar[2] : inverse scale (rate) parameter ("a")
    args : tuple
        Tuple of variables and bounds to pass to weight_from_thresh:
            args[0]: Data vector
            args[1]: Vector of standard deviations
            args[2]: Lower bound of the thresholds
            args[3]: Upper bound of the thresholds

    Returns
    -------
    -loglik : float
        The narginal negative log likelihood.
        
    """

    a = xpar[1]
    xx = args[0]
    ss = args[1]
    tlo = args[2]
    thi = args[3]
    
    # Calculate the range of weights given a scale parameter using negative
    # monotonicity between the weight and the threshold
    
    wlo = weight_from_thresh(thi, ss, prior='laplace', a=a)
    whi = weight_from_thresh(tlo, ss, prior='laplace', a=a)
    wlo = np.max(wlo)
    whi = np.min(whi)
    loglik = np.sum(np.log(1 + (xpar[0] * (whi - wlo) + wlo) * beta_laplace(xx,s=ss,a=a)))
    
    return -loglik
    
    