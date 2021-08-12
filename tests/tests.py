#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 16:03:51 2021

@author: artsloan
"""

import numpy as np
from pytest import approx
import e_bayes_thresh as ebt

#%% Test beta_laplace.py

x = np.array([-2,1,0,-4,5])
y_valid = np.array([+0.889852029651143,
                    -0.380041716606011,
                    -0.561817771773154,
                    +285.459466672351,
                    +15639.8849145429])
y = ebt.beta_laplace(x)

assert ebt.beta_laplace(x) == approx(y_valid,)

#%% Test beta_cauchy.py
x = np.array([-2,1,0,-4,5])
y_valid = np.array([+0.597264024732662,
                    -0.351278729299872,
                    -0.500000000000000,
                    +185.247374190108,
                    +10732.4514608350])

assert ebt.beta_cauchy(x) == approx(y_valid,)

#%% Test post_mean with the laplace prior
x = np.array([-2,1,0,-4,5])
y_valid = np.array([-1.01158962199946,
                    +0.270953305249239,
                    0.0,
                    -3.48800924041643,
                    +4.4997151290092])

assert ebt.post_mean(x,prior='laplace') == approx(y_valid,)

#%% Test post_mean with the cauchy prior
x = np.array([-2,1,0,-4,5])
y_valid = np.array([-0.807489729485063,
                    +0.213061319425267,
                    0.0,
                    -3.48264328130604,
                    +4.59959010481196])

assert ebt.post_mean(x,prior='cauchy') == approx(y_valid,)
#%% Test post_med with the laplace prior
x = np.array([-2,1,0,-4,5])
y_valid = np.array([-0.829992882781227,
                    0.0,
                    0.0,
                    -3.49568406354978,
                    +4.49992059554046])

assert ebt.post_med(x,prior='laplace') == approx(y_valid,)
#%% Test post_med with the cauchy prior
x = np.array([-2,1,0,-4,5])
y_valid = np.array([-0.398555518768262,
                    0.0,
                    0.0,
                    -3.50192235462600,
                    +4.60455881926464])

assert ebt.post_med(x,prior='cauchy')[0] == approx(y_valid,)

#%% Test thresh_from_weight with the laplace prior
w = np.arange(0.2,0.8,0.2)
y_valid = np.array([+2.44873377028853,
                    +1.92279064562172,
                    +1.40956187155098,
                    +0.767900790087879])

assert ebt.thresh_from_weight(w,prior='laplace')[0] == approx(y_valid)

#%% Test thresh_from_weight with the cauchy prior
w = np.arange(0.2,0.8,0.2)
y_valid = np.array([+2.60031945683295,
                    +2.05919773929054,
                    +1.51366172562120,
                    +0.818831556534860])

assert ebt.thresh_from_weight(w,prior='cauchy')[0] == approx(y_valid)

#%% Test thresh_from_data with the laplace prior
x = np.array([-0.560475647, -0.230177489, 1.558708314, 0.070508391, 0.129287735, 1.715064987, 
              0.460916206, -1.265061235, -0.686852852, -0.445661970, 1.224081797, 0.359813827,
              0.400771451, 0.110682716, -0.555841135, 1.786913137, 0.497850478, -1.966617157,
              0.701355902, -0.472791408, -1.067823706, -0.217974915, -1.026004448, -0.728891229,
              -0.625039268, -1.686693311, 0.837787044, 0.153373118, -1.138136937, 1.253814921, 
              0.426464221, -0.295071483, 0.895125661, 0.878133488, 0.821581082, 0.688640254, 
              0.553917654, -0.061911711, -0.305962664, -0.380471001, -0.694706979, -0.207917278, 
              -1.265396352, 2.168955965, 1.207961998, -1.123108583, -0.402884835, -0.466655354,
              0.779965118, -0.083369066, 0.253318514, -0.028546755, -0.042870457, 1.368602284,
              -0.225770986, 1.516470604, -1.548752804, 0.584613750, 0.123854244, 0.215941569,
              0.379639483, -0.502323453, -0.333207384, -1.018575383, -1.071791226, 0.303528641,
              0.448209779, 0.053004227, 0.922267468, 2.050084686, -0.491031166, -2.309168876,
              1.005738524, -0.709200763, -0.688008616, 1.025571370, -0.284773007, -1.220717712,
              0.181303480, -0.138891362, 0.005764186, 0.385280401, -0.370660032, 0.644376549,
              -0.220486562, 0.331781964, 1.096839013, 0.435181491, -0.325931586, 1.148807618,
              0.993503856, 0.548396960, 0.238731735, -0.627906076, 1.360652449, -0.600259587,
              2.187332993, 1.532610626, -0.235700359, -1.026420900])
y_valid = +3.03485425654799

assert ebt.thresh_from_data(x,prior='laplace')[0] == approx(y_valid)
#%% Test weight_from_thresh

#%% Test weight_from_data

#%% Test weifht_and_scale_from_data


