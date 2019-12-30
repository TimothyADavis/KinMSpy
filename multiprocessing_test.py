#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 18:14:17 2019

"""

import numpy as np
from astropy.convolution import convolve
from multiprocessing import Pool 
from itertools import repeat

psf = np.array([[0.1,0.1,0.1],[0.1,0.5,0.1],[0.1,0.1,0.1]])
cube = np.random.uniform(0,1,(120,600,600))
index = np.array([0,1,2,3,4,5,100])


def func(b, psf):
    print(b.shape, psf.shape)
    b = convolve(b,psf)
    return b

with Pool(10) as pool:
    L = pool.starmap(func, zip(cube,repeat(psf)))

    

    