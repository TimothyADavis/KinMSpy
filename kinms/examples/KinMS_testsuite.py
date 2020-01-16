# coding: utf-8
from kinms import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate,ndimage
from kinms.makebeam import makebeam
from kinms.examples.sauron_colormap import sauron
from astropy.io import fits
import time
import os.path
import socket

def internet(host="8.8.8.8", port=53, timeout=3):
  """
  Host: 8.8.8.8 (google-public-dns-a.google.com)
  OpenPort: 53/tcp
  Service: domain (DNS/TCP)
  """
  try:
    socket.setdefaulttimeout(timeout)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
    return True
  except socket.error as ex:
    print(ex)
    return False




def gaussian(x,x0,sigma):
  return np.exp(-np.power((x - x0)/(sigma), 2.)/2.)

def makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=0,overcube=False,pvdthick=2,nconts=11.,title=False, **kwargs):
    
# ;;;; Create plot data from cube ;;;;
    mom0rot=f.sum(axis=2)
    if np.any(overcube): mom0over=overcube.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*f[i,j,:]).sum()/f[i,j,:].sum()

    pvdcube=f 
    
    pvdcube=ndimage.interpolation.rotate(f, 90-posang, axes=(1, 0), reshape=False)
    if np.any(overcube): pvdcubeover=ndimage.interpolation.rotate(overcube, 90-posang, axes=(1, 0), reshape=False)
        
    pvd=pvdcube[:,np.int((ysize/(cellsize*2.))-pvdthick):np.int((ysize/(cellsize*2.))+pvdthick),:].sum(axis=1)
    if np.any(overcube): pvdover=pvdcubeover[:,np.int((ysize/(cellsize*2.))-pvdthick):np.int((ysize/(cellsize*2.))+pvdthick),:].sum(axis=1)
        
    if not isinstance(beamsize, (list, tuple, np.ndarray)):
        beamsize=np.array([beamsize,beamsize,0])
    beamtot=(makebeam(xsize,ysize,[beamsize[0]/cellsize,beamsize[1]/cellsize],rot=beamsize[2])).sum()
    spec=f.sum(axis=0).sum(axis=0)/beamtot
    if np.any(overcube): specover=overcube.sum(axis=0).sum(axis=0)/beamtot
     
# ;;;;

# ;;;; Plot the results ;;;;
    levs=v1[np.min(np.where(spec != 0)):np.max(np.where(spec != 0))]
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax1 = fig.add_subplot(221, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax1.contourf(x1,y1,mom0rot.T,levels=np.linspace(1,0,num=10,endpoint=False)[::-1]*np.max(mom0rot), cmap="YlOrBr")
    if np.any(overcube): ax1.contour(x1,y1,mom0over.T,colors=('black'),levels=np.arange(0.1, 1.1, 0.1)*np.max(mom0over))
    if 'yrange' in kwargs: ax1.set_ylim(kwargs['yrange'])
    if 'xrange' in kwargs: ax1.set_xlim(kwargs['xrange'])
    ax2 = fig.add_subplot(222, aspect='equal')
    plt.xlabel('Offset (")')
    plt.ylabel('Offset (")')
    ax2.contourf(x1,y1,mom1.T,levels=levs, cmap=sauron)
    if 'yrange' in kwargs: ax2.set_ylim(kwargs['yrange'])
    if 'xrange' in kwargs: ax2.set_xlim(kwargs['xrange'])
    ax3 = fig.add_subplot(223)
    plt.xlabel('Offset (")')
    plt.ylabel(r'Velocity (km s$^{-1}$)')
    ax3.contourf(x1,v1,pvd.T,levels=np.linspace(1,0,num=10,endpoint=False)[::-1]*np.max(pvd), cmap="YlOrBr" ,aspect='auto')
    if np.any(overcube): ax3.contour(x1,v1,pvdover.T,colors=('black'),levels=np.arange(0.1, 1.1, 0.1)*np.max(pvdover))
    if 'vrange' in kwargs: ax3.set_ylim(kwargs['vrange'])
    if 'xrange' in kwargs: ax3.set_xlim(kwargs['xrange'])
    ax4 = fig.add_subplot(224)
    plt.ylabel('Flux')
    plt.xlabel(r'Velocity (km s$^{-1}$)')
    ax4.plot(v1,spec, drawstyle='steps')
    if np.any(overcube): ax4.plot(v1,specover,'r', drawstyle='steps')
    if 'vrange' in kwargs: ax4.set_xlim(kwargs['vrange'])
    if title: plt.suptitle(title)
    plt.show()
# ;;;;




def KinMStest_expdisk(scalerad=10.,inc=45.,fileName=None):
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to create a simulation of an
# ; exponential disk of molecular gas. The user can input values for the
# ; scalerad and inc variables, and the procedure will the create the simulation
# ; and display it to screen. 
# ;
# ;  INPUTS:
# ;       Scalerad -    Scale radius for the exponential disk (arcseconds)
# ;       Inc      -    Inclination to project the disk (degrees).
# ;
# ;;;;;;;;;;;

# ;;;; Setup cube parameters ;;;;
    xsize=128
    ysize=128
    vsize=1400
    cellsize=1
    dv=10
    beamsize=[4.,4.,0.]
    nsamps=5e5
    v_flat=210.
    r_turn=2.0

# ;;;; Set up exponential disk SB profile/velocity ;;;;
    x=np.arange(0,100,0.1)
    fx = np.exp(-x/scalerad)
    # velfunc = interpolate.interp1d([0.0,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=(v_flat*2/np.pi)*np.arctan(x/r_turn)
# ;;;;

# ;;;; Simulate ;;;;
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=270,gasSigma=10.,fileName=None)

# ;;;; Plot
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=270.)



def KinMStest_expdisk_gasgrav(scalerad=5.,inc=45.,gasmass=5e10):
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to create a simulation of an
# ; exponential disk of molecular gas, including the effect of the potential of the gas on its own rotation.
# ; The user can input values for the scalerad and inc variables, and the procedure will 
# ; the create the simulation and display it to screen. 
# ;
# ;  INPUTS:
# ;       Scalerad -    Scale radius for the exponential disk (arcseconds)
# ;       Inc      -    Inclination to project the disk (degrees)
# ;       Gasmass  -    Total mass of the gas (solar masses)
# ;
# ;;;;;;;;;;;

# ;;;; Setup cube parameters ;;;;
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
    plt.ion()
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=270,gasSigma=10.)
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=270.,title="Without Potential of Gas")
        

    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=270,gasSigma=10.,gasGrav=np.array([gasmass,dist]))
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=270.,title="With Potential of Gas Included")


    

def KinMStest_ngc4324():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure makes a basic simulation of the
# ; molecular gas ring in NGC4324, and plots the simulation moment zero,
# ; one and PVD against the observed ones from the CARMA observations
# ; of Alatalo et al., 2012. 
# ;
# ;;;;;;;;;;;

# ;;; Define the simulated observation parameters ;;;
    xsize=100. # arcseconds
    ysize=100. #;; arcseconds
    vsize=420. #;; km/s
    cellsize=1. #;; arcseconds/pixel
    dv=20. #;; km/s/channel
    beamsize=np.array([4.68,3.85,15.54]) #;; arcseconds
# ;;;
#
# ;;; Define the gas distribution required ;;;
    diskthick=1. # arcseconds
    inc=65. # degrees
    posang=230. # degrees
    x=np.arange(0.,64.)
    fx = 0.1*gaussian(x,20.0,2.0)
    velfunc = interpolate.interp1d([0.0,1,3,5,7,10,200],[0,50,100,175,175,175,175], kind='linear')
    vel=velfunc(x)
    phasecen=[-1,-1]
    voffset=0
# ;;;

# ;;; Run KinMS
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=5e5,posAng=posang,intFlux=27.2,phaseCen=phasecen,vOffset=voffset,gasSigma=10.,fileName="NGC4234_test")
# ;;;

# ;;; Read in data

    hdulist = fits.open('http://www.astro.cardiff.ac.uk/pub/Tim.Davis/NGC4324.fits')
    scidata = hdulist[0].data.T
    scidata=scidata[:,:,:,0]
    scidata[np.where(scidata < np.std(scidata[:,:,0])*4.)]=0.0
# ;;;

# ;;; Create the plot 
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=posang,overcube=scidata,xrange=[-28,28],yrange=[-28,28],pvdthick=4.)

def KinMStest_inclouds():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to use the
# ; INCLOUDS parameter set to create simulations, in this case of a very
# ; unrealistic object. Once you understand this example then see the
# ; INFITS and INCLOUDS_SPIRAL test for more realistic examples.
# ;
# ;;;;;;;;;;;

# ;;;; Setup cube parameters ;;;;
    xsize=128
    ysize=128
    vsize=1400
    cellsize=1.0
    dv=10
    beamsize=[4.,4.,0]
    inc=35.
    x=np.arange(0.,50,0.1)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
# ;;;;

# ;;;; define where clouds are in each dimension (x,y,z) ;;;;
    inclouds=np.array([[ 40.0000, 0.00000, 0.00000],[ 39.5075, 6.25738, 0.00000],[ 38.0423, 12.3607, 0.00000],[ 35.6403, 18.1596, 0.00000],[ 32.3607, 23.5114, 0.00000],[ 28.2843, 28.2843, 0.00000],[ 23.5114, 32.3607, 0.00000],[ 18.1596, 35.6403, 0.00000],[ 12.3607, 38.0423, 0.00000],[ 6.25737, 39.5075, 0.00000],[ 0.00000, 40.0000, 0.00000],[-6.25738, 39.5075, 0.00000],[-12.3607, 38.0423, 0.00000],[-18.1596, 35.6403, 0.00000],[-23.5114, 32.3607, 0.00000],[-28.2843, 28.2843, 0.00000],[-32.3607, 23.5114, 0.00000],[-35.6403, 18.1596, 0.00000],[-38.0423, 12.3607, 0.00000],[-39.5075, 6.25738, 0.00000],[-40.0000, 0.00000, 0.00000],[-39.5075,-6.25738, 0.00000],[-38.0423,-12.3607, 0.00000],[-35.6403,-18.1596, 0.00000],[-32.3607,-23.5114, 0.00000],[-28.2843,-28.2843, 0.00000],[-23.5114,-32.3607, 0.00000],[-18.1596,-35.6403, 0.00000],[-12.3607,-38.0423, 0.00000],[-6.25738,-39.5075, 0.00000],[ 0.00000,-40.0000, 0.00000],[ 6.25738,-39.5075, 0.00000],[ 12.3607,-38.0423, 0.00000],[ 18.1596,-35.6403, 0.00000],[ 23.5114,-32.3607, 0.00000],[ 28.2843,-28.2843, 0.00000],[ 32.3607,-23.5114, 0.00000],[ 35.6403,-18.1596, 0.00000],[ 38.0423,-12.3607, 0.00000],[ 39.5075,-6.25737, 0.00000],[ 15.0000, 15.0000, 0.00000],[-15.0000, 15.0000, 0.00000],[-19.8504,-2.44189, 0.00000],[-18.0194,-8.67768, 0.00000],[-14.2856,-13.9972, 0.00000],[-9.04344,-17.8386, 0.00000],[-2.84630,-19.7964, 0.00000],[ 3.65139,-19.6639, 0.00000],[ 9.76353,-17.4549, 0.00000],[ 14.8447,-13.4028, 0.00000],[ 18.3583,-7.93546, 0.00000],[ 19.9335,-1.63019, 0.00000]])
# ;;;;

# ;;;; run the simulation with a velocity curve ;;;;
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,intFlux=30.,inClouds=inclouds,velProf=vel,velRad=x,posAng=90.)
# ;;;;
# ;;; Create the plot 

    mom0rot=f.sum(axis=2)
    x1=np.arange(-xsize/2.,xsize/2.,cellsize)
    y1=np.arange(-ysize/2.,ysize/2.,cellsize)
    v1=np.arange(-vsize/2.,vsize/2.,dv)

    mom1=(mom0rot*0.0)-10000.0
    for i in range(0,int(xsize/cellsize)):
         for j in range(0,int(ysize/cellsize)):
             if mom0rot[i,j] > 0.1*np.max(mom0rot):
                 mom1[i,j]=(v1*f[i,j,:]).sum()/f[i,j,:].sum()
                 
    levs=v1[np.where(f.sum(axis=0).sum(axis=0))]
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

def KinMStest_inclouds_spiral():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to use the
# ; INCLOUDS parameter set to create simulations, in this case of
# ; molecular gas in a two armed spiral pattern. Any default paramaters
# ; can be changed by specifying them at the command line (see KinMS.pro
# ; for the full details of all the avaliable parameters).
# ;
# ;
# ;;;;;;;;;;;

# ;;;; Setup cube parameters ;;;;
    xsize=128
    ysize=128
    vsize=600
    cellsize=1
    dv=10
    beamsize=4.
    inc=55.
# ;;;;
#
# ;;;; define where clouds are in each dimension (x,y,z) using a logarithmic spiral ;;;;
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
# ;;;;
    inclouds=inclouds[np.where((abs(inclouds[:,0]) > 2.0) & (abs(inclouds[:,1]) > 2.0))[0],:]

# ;;;; define velocity curve ;;;;
    x=np.arange(0.,5000)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,5000],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
# ;;;;

# ;;;; run the simulation ;;;;
    cube=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,intFlux=30.,inClouds=inclouds,velProf=vel,velRad=x,posAng=90)
# ;;;;

# ;;;; Plot the results ;;;;
    plot=makeplots(cube,xsize,ysize,vsize,cellsize,dv,beamsize,pvdthick=50.)
# ;;;;


def KinMStest_infits():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to use the
# ; an input FITS image to create a simulation of what the molecular gas
# ; may look like, with a given instrument (in this case CARMA). We use a
# ; GALEX (Morrissey et al., 2007) FUV image of NGC1437A, and scale it
# ; assuming the FUV emission comes from star-formation and thus
# ; molecular gas, and that the galaxy has a total integrated CO flux of
# ; 30 Jy km/s. We use the FITS image to set the surface-brightness, and
# ; impose a flat velocity gradient.
# ;
# ;;;;;;;;;;;
#
# ;;;; Setup cube parameters ;;;;
    xsize=128
    ysize=128
    vsize=500
    cellsize=1.0
    dv=10
    beamsize=4.
    inc=0.
# ;;;;
#
#
# ;;;; Read in the FITS file and create the INCLOUDS variables based on it ;;;;
    phasecent=[88,61] # point we wish to correspond to the phase centre in the simulation
    hdulist = fits.open('http://www.astro.cardiff.ac.uk/pub/Tim.Davis/NGC1437A_FUV.fits')
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

# ;;;;
#
# ;;;; run the simulation ;;;;
    
    cube=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,intFlux=30.,inClouds=inclouds,vLOS_clouds=vlos_clouds,flux_clouds=flux_clouds)
# ;;;; Plot the results ;;;;
    plot=makeplots(cube,xsize,ysize,vsize,cellsize,dv,beamsize)
# ;;;;






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
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,gasSigma=gassigma,posAng=90)
# ;;;;

# ;;;; Plot
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,vrange=[-200,200],posang=90)


def KinMStest_diskthick():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to create a simulation of an
# ; exponential disk of molecular gas with a thickness that
# ; varies with radius. Any default paramaters
# ; can be changed by specifying them at the command line (see KinMS.pro
# ; for the full details of all the avaliable parameters).
# ;
# ;
# ;;;;;;;;;;;

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
    inc=90.
    x=np.arange(0,100,0.1)
    fx = fcent*np.exp(-x/scalerad)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    diskthickfunc = interpolate.interp1d([0,10,15,20,200],[1,1,5,15,15], kind='linear')
    diskthick=diskthickfunc(x)
# ;;;;

# ;;;; Simulate ;;;;
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,diskThick=diskthick,posAng=270)
# ;;;;

# ;;;; Plot
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,vrange=[-250,250],posang=90,xrange=[-30,30],yrange=[-30,30])


def KinMStest_warp():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to create a simulation of a
# ; warped exponential disk of molecular gas. 
# ;
# ;;;;;;;;;;;

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
    inc=60.
    x=np.arange(0,100,0.1)
    fx = fcent*np.exp(-x/scalerad)
    velfunc = interpolate.interp1d([0.00,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    diskthickfunc = interpolate.interp1d([0.0,15,50,500],[270,270,300,300], kind='linear')
    posang=diskthickfunc(x)
# ;;;;

# ;;;; Simulate ;;;;
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=posang)
# ;;;;

# ;;;; Plot
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,vrange=[-250,250],posang=270)

def KinMStest_retclouds():
# ;;;;;;;;;;;
# ;
# ; A test procedure to demonstrate the KinMS code, and check if it
# ; works on your system. This procedure demonstrates how to use the
# ; return clouds feature to recursivly build models - here a 
# ; misaligned central and outer disc.
# ;
# ;;;;;;;;;;;

# ;;;; Setup cube parameters ;;;;
    xsize=64.
    ysize=64.
    vsize=1000
    cellsize=1
    dv=10
    beamsize=[4.,4.,0.]


# ;;;; Set up exponential disk SB profile/velocity for disc one ;;;;
    inc=75.
    x=np.arange(0,100,0.1)
    fx = np.exp(-x/4.)
    fx[np.where(x > 5.)]=0.0
    velfunc = interpolate.interp1d([0.0,0.5,1,3,500],[0,50,100,210,210], kind='linear')
    vel=velfunc(x)
    nsamps=5e4
# ;;;;

# ;;;; Simulate disc 1 ;;;;
    __,inclouds1,vlos1=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=90,gasSigma=10.,returnClouds=True)

# ;;;; Set up exponential disk SB profile for disc two ;;;;
    inc=35.
    x=np.arange(0,100,0.1)
    fx = np.exp(-x/15.)
    fx[np.where(x < 10.)]=0.0
    nsamps=1e6
# ;;;;

# ;;;; Simulate disc 2 ;;;;
    __,inclouds2,vlos2=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,sbProf=fx,sbRad=x,velRad=x,velProf=vel,nSamps=nsamps,intFlux=30.,posAng=270,gasSigma=10.,returnClouds=True)

# ;;;; Combine ;;;;

    inclouds=np.concatenate((inclouds1,inclouds2),axis=0)
    vlos=np.concatenate((vlos1,vlos2))

# ;;;; Simulate whole thing ;;;;
    
    f=KinMS(xsize,ysize,vsize,cellsize,dv,beamsize,inc,inClouds=inclouds,vLOS_clouds=vlos,intFlux=30.)
    

# ;;;; Plot
    plot=makeplots(f,xsize,ysize,vsize,cellsize,dv,beamsize,posang=270.)


def run_tests():
    have_connection=internet()
    if have_connection: 
        print("Test - simulate the gas ring in NGC4324")
        print("[Close plot to continue]")
        KinMStest_ngc4324()
    print("Test - simulate an exponential disk")
    print("[Close plot to continue]")
    KinMStest_expdisk()
    print("Test - using the INCLOUDS mechanism - unrealistic")
    print("[Close plot to continue]")
    KinMStest_inclouds()
    print("Test - using the INCLOUDS mechanism - realistic")
    print("[Close plot to continue]")
    KinMStest_inclouds_spiral()
    if have_connection: 
        print("Test - using a FITS file as input")
        print("[Close plot to continue]")
        KinMStest_infits()
    print("Test - using variable velocity dispersion")
    print("[Close plot to continue]")
    KinMStest_veldisp()
    print("Test - using variable disk thickness")
    print("[Close plot to continue]")
    KinMStest_diskthick()
    print("Test - simulate a warped exponential disk")
    print("[Close plot to finish]")
    KinMStest_warp()
    print("Test - using the returnclouds mechanism")
    print("[Close plot to finish]")
    KinMStest_retclouds()
    print("Test - using the gravgas mechanism")
    KinMStest_expdisk_gasgrav()
    