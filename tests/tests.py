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
                    0,
                    -3.48800924041643,
                    +4.4997151290092])

assert ebt.post_mean(x,prior='laplace') == approx(y_valid,)




#%% Test post_mean with the cauchy prior
x = np.array([-2,1,0,-4,5])
y_valid = np.array([-0.807489729485063,
                    +0.213061319425267,
                    0,
                    -3.48264328130604,
                    +4.59959010481196])

assert ebt.post_mean(x,prior='cauchy') == approx(y_valid,)