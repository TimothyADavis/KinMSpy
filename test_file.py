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

from  astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

from TimMS import KinMS
from scipy import interpolate
from sauron_colormap import sauron

import time
from tqdm import tqdm

#=============================================================================#
#/// TESTY TEST //////////////////////////////////////////////////////////////#
#=============================================================================#

def exp_disk():

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
        inc= 60
        
        kinms = KinMS()
        cube = kinms(xsize,ysize,vsize,cellsize,dv,beamSize=beamsize,inc=inc,sbProf=fx,sbRad=x,velProf=vel,velRad=x,posAng=pos,verbose=False)
        flattened = cube.sum(axis=2)
        plt.figure()
        plt.imshow(flattened,cmap='inferno')
        
#exp_disk()

#=============================================================================#
#/// WITH GASGRAV  ///////////////////////////////////////////////////////////#
#=============================================================================#
        
start =  time.time()
        
def gasgrav():

        scalerad=5.
        inc=45.
        gasmass=5e10
        xsize=64
        ysize=64
        vsize=1400
        cellsize=1
        dv=10
        beamsize=[4.,4.,0.]
        nsamps=5e5
        dist=16.5 # Mpc
        
        # ;;;; Set up exponential disk SB profile/velocity ;;;;
        x=np.arange(0,100,0.1)
        fx = np.exp(-x/scalerad)
        velfunc = interpolate.interp1d([0.0,0.5,1,3,500],[0,50,100,210,210], kind='linear')
        vel=velfunc(x)
        # ;;;;
        
        # ;;;; Simulate and plot ;;;;
        cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,
                   velProf=vel,nSamps=nsamps,intFlux=30.,posAng=270,gasSigma=10.,verbose=False)
        
#        mom0rot=cube.sum(axis=2)
#        x1=np.arange(-xsize/2.,xsize/2.,cellsize)
#        y1=np.arange(-ysize/2.,ysize/2.,cellsize)
#        v1=np.arange(-vsize/2.,vsize/2.,dv)
#        
#        mom1=(mom0rot*0.0)-10000.0
#        for i in range(0,int(xsize/cellsize)):
#             for j in range(0,int(ysize/cellsize)):
#                 if mom0rot[i,j] > 0.1*np.max(mom0rot):
#                     mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
#                         
#        levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
#        fig = plt.figure()
#        fig.patch.set_facecolor('white')
#        ax1 = fig.add_subplot(121, aspect='equal')
#        plt.xlabel('Offset (")')
#        plt.ylabel('Offset (")')
#        ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
#        ax2 = fig.add_subplot(122, aspect='equal')
#        plt.xlabel('Offset (")')
#        plt.ylabel('Offset (")')
#        ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
#        plt.show()                 
        
        cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,
                   velProf=vel,nSamps=nsamps,intFlux=30.,posAng=270,gasSigma=10.,gasGrav=[gasmass,dist],verbose=False)
        
#        mom0rot=cube.sum(axis=2)
#        x1=np.arange(-xsize/2.,xsize/2.,cellsize)
#        y1=np.arange(-ysize/2.,ysize/2.,cellsize)
#        v1=np.arange(-vsize/2.,vsize/2.,dv)
#        
#        mom1=(mom0rot*0.0)-10000.0
#        for i in range(0,int(xsize/cellsize)):
#             for j in range(0,int(ysize/cellsize)):
#                 if mom0rot[i,j] > 0.1*np.max(mom0rot):
#                     mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
#                         
#        levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
#        fig = plt.figure()
#        fig.patch.set_facecolor('white')
#        ax1 = fig.add_subplot(121, aspect='equal')
#        plt.xlabel('Offset (")')
#        plt.ylabel('Offset (")')
#        ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
#        ax2 = fig.add_subplot(122, aspect='equal')
#        plt.xlabel('Offset (")')
#        plt.ylabel('Offset (")')
#        ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
#        plt.show() 
 
#kinms = KinMS()
#for _ in tqdm(range(1000)):
#        gasgrav()
#        
#end = time.time()
#print(end-start)

#=============================================================================#
#/// INCLLOUDS TEST  /////////////////////////////////////////////////////////#
#=============================================================================#


        
def inclouds():

    xsize=128
    ysize=128
    vsize=1400
    cellsize=1.0
    dv=10
    beamsize=[4.,4.,0]
    inc=35.
    x=np.arange(0.,100,0.1)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)

    inclouds=np.array([[ 40.0000, 0.00000, 0.00000],[ 39.5075, 6.25738, 0.00000],[ 38.0423, 12.3607, 0.00000],[ 35.6403, 18.1596, 0.00000],[ 32.3607, 23.5114, 0.00000],[ 28.2843, 28.2843, 0.00000],[ 23.5114, 32.3607, 0.00000],[ 18.1596, 35.6403, 0.00000],[ 12.3607, 38.0423, 0.00000],[ 6.25737, 39.5075, 0.00000],[ 0.00000, 40.0000, 0.00000],[-6.25738, 39.5075, 0.00000],[-12.3607, 38.0423, 0.00000],[-18.1596, 35.6403, 0.00000],[-23.5114, 32.3607, 0.00000],[-28.2843, 28.2843, 0.00000],[-32.3607, 23.5114, 0.00000],[-35.6403, 18.1596, 0.00000],[-38.0423, 12.3607, 0.00000],[-39.5075, 6.25738, 0.00000],[-40.0000, 0.00000, 0.00000],[-39.5075,-6.25738, 0.00000],[-38.0423,-12.3607, 0.00000],[-35.6403,-18.1596, 0.00000],[-32.3607,-23.5114, 0.00000],[-28.2843,-28.2843, 0.00000],[-23.5114,-32.3607, 0.00000],[-18.1596,-35.6403, 0.00000],[-12.3607,-38.0423, 0.00000],[-6.25738,-39.5075, 0.00000],[ 0.00000,-40.0000, 0.00000],[ 6.25738,-39.5075, 0.00000],[ 12.3607,-38.0423, 0.00000],[ 18.1596,-35.6403, 0.00000],[ 23.5114,-32.3607, 0.00000],[ 28.2843,-28.2843, 0.00000],[ 32.3607,-23.5114, 0.00000],[ 35.6403,-18.1596, 0.00000],[ 38.0423,-12.3607, 0.00000],[ 39.5075,-6.25737, 0.00000],[ 15.0000, 15.0000, 0.00000],[-15.0000, 15.0000, 0.00000],[-19.8504,-2.44189, 0.00000],[-18.0194,-8.67768, 0.00000],[-14.2856,-13.9972, 0.00000],[-9.04344,-17.8386, 0.00000],[-2.84630,-19.7964, 0.00000],[ 3.65139,-19.6639, 0.00000],[ 9.76353,-17.4549, 0.00000],[ 14.8447,-13.4028, 0.00000],[ 18.3583,-7.93546, 0.00000],[ 19.9335,-1.63019, 0.00000]])

    kinms = KinMS()        
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,intFlux=30.,inClouds=inclouds,velProf=vel,velRad=x,posAng=90.)

#    mom0rot=cube.sum(axis=2)
#    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
#    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
#    v1=np.arange(-vsize/2.,vsize/2.,dv)
#
#    mom1=(mom0rot*0.0)-10000.0
#    for i in range(0,int(xsize/cellsize)):
#         for j in range(0,int(ysize/cellsize)):
#             if mom0rot[i,j] > 0.1*np.max(mom0rot):
#                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
#                 
#    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
#    fig = plt.figure()
#    fig.patch.set_facecolor('white')
#    ax1 = fig.add_subplot(121, aspect='equal')
#    plt.xlabel('Offset (")')
#    plt.ylabel('Offset (")')
#    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
#    ax2 = fig.add_subplot(122, aspect='equal')
#    plt.xlabel('Offset (")')
#    plt.ylabel('Offset (")')
#    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
#    plt.show()        
 
start =  time.time()
for _ in tqdm(range(100)):
        inclouds()
end = time.time()
print(end-start)

### Time = 10.686s

#=============================================================================#
#/// SPIRAL TEST  ////////////////////////////////////////////////////////////#
#=============================================================================#

def spiral():

    xsize=128
    ysize=128
    vsize=1400
    cellsize=1
    dv=10
    beamsize=4.
    inc=55.

    t=np.arange(-20,20,0.1)
    a=0.002*60
    b=0.5
    x1=a*np.exp(b*t)*np.cos(t)
    y1=a*np.exp(b*t)*np.sin(t)
    x2=a*(-1)*np.exp(b*t)*np.cos(t)
    y2=a*(-1)*np.exp(b*t)*np.sin(t)
    inclouds=np.empty((len(t)*2,3))
    inclouds[:,0]=np.concatenate((x1,x2))/20
    inclouds[:,1]=np.concatenate((y1,y2))/20
    inclouds[:,2]=np.concatenate((x1*0.0,x2*0.0))

    inclouds=inclouds[np.where((abs(inclouds[:,0]) > 2.0) & (abs(inclouds[:,1]) > 2.0))[0],:]

    x=np.arange(0.,5000)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,5000],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    
    kinms = KinMS()
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,intFlux=30.,inClouds=inclouds,velProf=vel,velRad=x,posAng=90)

    mom0rot=cube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
                 
    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    plt.show() 
       
#spiral()


#=============================================================================#
#/// NGC1437A TEST  //////////////////////////////////////////////////////////#
#=============================================================================#
    
def KinMStest_infits():

    xsize=128
    ysize=128
    vsize=500
    cellsize=1.0
    dv=10
    beamsize=4.
    inc=0.

    phasecent=[88,61] # point we wish to correspond to the phase centre in the simulation
    hdulist = fits.open('test_suite/NGC1437A_FUV.fits')
    fin = hdulist[0].data.T
    s=fin.shape
    

    xvec=np.arange(0-phasecent[0], s[0]-phasecent[0])*(hdulist[0].header['cdelt1']*3600.)
    yvec=np.arange(0-phasecent[1], s[1]-phasecent[1])*(hdulist[0].header['cdelt2']*3600.)
    w=np.where(fin > 0.002) # clip the image to avoid FUV noise entering the simulation
    flux_clouds=fin[w]
    x=xvec[w[0]]
    y=yvec[w[1]]
    inclouds=np.empty((x.size,3))
    inclouds[:,0]=x
    inclouds[:,1]=y
    inclouds[:,2]=x*0.0
    
    ang=np.radians(80.)
    velfunc = interpolate.interp1d([-130,0,130],[-400,0,400], kind='linear')

    vlos_clouds=velfunc(y*np.sin(ang)+x*np.cos(ang)) # impose a flat velocity profile

    kinms = KinMS()
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,intFlux=30.,inClouds=inclouds,vLOS_clouds=vlos_clouds,flux_clouds=flux_clouds)
    
    mom0rot=cube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
                 
    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    plt.show() 
    
#KinMStest_infits()

#=============================================================================#
#/// Veldisp TEST  ///////////////////////////////////////////////////////////#
#=============================================================================#

def KinMStest_veldisp():
# ;;;;;;;;;;;
#;
#; A test procedure to demonstrate the KinMS code, and check if it
#; works on your system. This procedure demonstrates how to create a simulation of an
#; exponential disk of molecular gas with a velocity dispersion that
#; varies with radius.
#;
#;;;;;;;;;;;
# ;;;; Setup cube parameters ;;;;
    xsize=128
    ysize=128
    vsize=1400
    cellsize=1.0
    dv=10
    beamsize=2.
    nsamps=5e5
# ;;;;

# ;;;; Set up exponential disk SB profile/velocity ;;;;
    fcent=10.
    scalerad=20.
    inc=30.
    x=np.arange(0,100,0.1)
    fx = fcent*np.exp(-x/scalerad)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    gassigfunc = interpolate.interp1d([0,20,500],[50,8,8], kind='linear')
    gassigma=gassigfunc(x)
# ;;;;

# ;;;; Simulate ;;;;
    kinms = KinMS()
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,gasSigma=gassigma,posAng=90)    
    
    mom0rot=cube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
                 
    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    plt.show() 
    
#KinMStest_veldisp()

#=============================================================================#
#/// diskthick TEST  /////////////////////////////////////////////////////////#
#=============================================================================#
    
def KinMStest_diskthick():

    xsize=128
    ysize=128
    vsize=1400
    cellsize=1.0
    dv=10
    beamsize=2.
    nsamps=5e5

    fcent=10.
    scalerad=20.
    inc=90.
    x=np.arange(0,100,0.1)
    fx = fcent*np.exp(-x/scalerad)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    diskthickfunc = interpolate.interp1d([0,10,15,20,200],[1,1,5,15,15], kind='linear')
    diskthick=diskthickfunc(x)

    kinms = KinMS()
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,diskThick=diskthick,posAng=270)
    
    mom0rot=cube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
                 
    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    plt.show() 

#KinMStest_diskthick()    
    
#=============================================================================#
#/// warp TEST  //////////////////////////////////////////////////////////////#
#=============================================================================#

def KinMStest_warp():

    xsize=128
    ysize=128
    vsize=1400
    cellsize=1.0
    dv=10
    beamsize=2.
    nsamps=5e5

    fcent=10.
    scalerad=20.
    inc=60.
    x=np.arange(0,100,0.1)
    fx = fcent*np.exp(-x/scalerad)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    diskthickfunc = interpolate.interp1d([0.0,15,50,500],[270,270,300,300], kind='linear')
    posang=diskthickfunc(x)

    kinms = KinMS()
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=posang,verbose=True)    
    
    mom0rot=cube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
                 
    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    plt.show() 

#KinMStest_warp()
    
#=============================================================================#
#/// warp TEST 2  ////////////////////////////////////////////////////////////#
#=============================================================================#

def KinMStest_warp2():

    xsize=128
    ysize=128
    vsize=1400
    cellsize=1.0
    dv=10
    beamsize=2.
    nsamps=5e5

    fcent=10.
    scalerad=20.
    x=np.arange(0,100,0.1)
    fx = fcent*np.exp(-x/scalerad)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    diskthickfunc = interpolate.interp1d([0.0,15,50,250,500],[0,40,90,90,90], kind='linear')
    posang = 45
    inc=diskthickfunc(x)

    kinms = KinMS()
    cube=kinms(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=posang,verbose=True)    
    
    mom0rot=cube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*cube[i,j,:]).sum()/cube[i,j,:].sum()
                 
    levs=v1[np.where(cube.sum(axis=0).sum(axis=0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(121, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0rot), cmap="YlOrBr")
    ax2 = fig.add_subplot(122, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    plt.show() 

#KinMStest_warp2()

#=============================================================================#
#/// END OF SCRIPT ///////////////////////////////////////////////////////////#
#=============================================================================#
    
