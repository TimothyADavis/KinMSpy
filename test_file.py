#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (C) 2016, Timothy A. Davis
E-mail: DavisT -at- cardiff.ac.uk

Updated versions of the software are available through github:
https://github.com/TimothyADavis/KinMSpy

If you have found this software useful for your research,
I would appreciate an acknowledgment to the use of the
"KINematic Molecular Simulation (KinMS) routines of Davis et al., (2013)".
[MNRAS, Volume 429, Issue 1, p.534-555]

This software is provided as is without any warranty whatsoever.
For details of permissions granted please see LICENCE.md
"""
#=============================================================================#
#/// IMPORT PACKAGES /////////////////////////////////////////////////////////#
#=============================================================================#

import matplotlib.pyplot as plt
import numpy as np

from KinMS import KinMS

#=============================================================================#
#/// TESTY TEST //////////////////////////////////////////////////////////////#
#=============================================================================#

extent = 64
scale_length = extent/4

x = np.arange(0,extent,0.01)
fx = np.exp(-x/scale_length)
xsize=128
ysize=128
vsize=640
cellsize=1.0
dv=10
beamsize=[4.,4.,0.]
vel = np.sqrt(x)
pos = 5
inc= 75

kinms = KinMS()
cube = kinms(xsize,ysize,vsize,cellsize,dv,beamSize=beamsize,inc=inc,sbProf=fx,sbRad=x,velProf=vel,velRad=x,posAng=pos,verbose=False)
flattened = cube.sum(axis=2)
plt.figure()
plt.imshow(flattened,cmap='inferno')

#=============================================================================#
#/// END OF SCRIPT ///////////////////////////////////////////////////////////#
#=============================================================================#