# coding: utf-8
"""
Copyright (C) 2019, Timothy A. Davis, Nikki Zabel, James M. Dawson
E-mail: DavisT -at- cardiff.ac.uk, zabelnj -at- cardiff.ac.uk, dawsonj5 -at- cardiff.ac.uk
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

from kinms import KinMS
from kinms.utils.KinMS_figures import KinMS_plotter
import numpy as np
from scipy import interpolate
from astropy.io import fits
import warnings; warnings.filterwarnings("ignore")
import socket
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

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


def expdisk(scalerad=10, inc=60):
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates how
    to create a simulation of an exponential disk of molecular gas. The user can input values for the scalerad and inc
    variables, and the procedure will the create the simulation and display it to screen.
    :param scalerad: Scale radius for the exponential disk (in arcseconds)
    :param inc: Inclination to project the disk (in degrees)
    :return: N/A
    """

    # Set up the cube parameters
    xsize = 64
    ysize = 64
    vsize = 1400
    cellsize = 1
    dv = 10
    beamsize = [4, 4, 0]
    posang = 90

    # Set up exponential disk SB profile/velocity
    x = np.arange(0, 100, 0.1)
    fx = np.exp(-x / scalerad)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel, intFlux=30,
                 posAng=posang, gasSigma=10, huge_beam=False).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysa.sum(axis=0).Tize, vsize, cellsize, dv, beamsize, posang=posang, savepath='.', savename='exp_disk',
    #              pdf=True).makeplots()
    
    return cube


def expdisk_gasgrav(scalerad=5, inc=45, gasmass=1e10, distance=16.5):
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to create a simulation of an exponential disk of molecular gas, including the effect of the potential of the gas
    on its own rotation. The user can input values for the scalerad and inc variables, and the procedure will the create
    the simulation and display it to screen.
    :param scalerad: Scale radius for the exponential disk (in arcseconds)
    :param inc: Inclination to project the disk (in degrees)
    :param gasmass: Total mass of the gas (in solar masses)
    :param distance: Distance to the galaxy (in Mpc)
    :return: N/A
    """

    # Set up cube parameters
    xsize = 64
    ysize = 64
    vsize = 800
    cellsize = 1
    dv = 10
    beamsize = [4, 4, 0]
    posang = 270
    intflux = 30
    gassig = 10

    # Set up exponential disk SB profile/velocity
    x = np.arange(0, 100, 0.1)
    fx = np.exp(-x / scalerad)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)

    # Create the cube WITHOUT gasgrav
    cube1 = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel,
                  intFlux=intflux, posAng=posang, gasSigma=gassig).model_cube()

    # Create the cube WITH gasgrav
    cube2 = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel,
                  intFlux=intflux, posAng=posang, gasSigma=gassig, massDist=[gasmass, distance])\
                    .model_cube()

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    KinMS_plotter(cube1, xsize, ysize, vsize, cellsize, dv, beamsize, posang=posang,overcube=cube2).makeplots()

    
    return cube1, cube2

def ngc4324():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure makes a basic
    simulation of the molecular gas ring in NGC4324, and plots the simulation moment zero, one and PVD against the
    observed ones from the CARMA observations of Alatalo et al., 2012.
    :return: N/A
    """

    # Define the simulated observation parameters
    xsize = 100  # arcseconds
    ysize = 100  # arcseconds
    vsize = 420  # km/s
    cellsize = 1  # arcseconds/pixel
    dv = 20  # km/s/channel
    beamsize = np.array([4.68, 3.85, 15.54])  # arcseconds

    #Define the gas distribution required
    diskthick = 1  # arcseconds
    inc = 65  # degrees
    posang = 230  # degrees
    intflux = 27.2
    gassigma = 10
    x = np.arange(0, 64)
    fx = 0.1 * np.exp(-np.power((x - 20) / 2, 2) / 2)
    velfunc = interpolate.interp1d([0, 1, 3, 5, 7, 10, 200], [0, 50, 100, 175, 175, 175, 175], kind='linear')
    vel = velfunc(x)
    phasecen = [-1, -1]
    voffset = 0

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel, posAng=posang, diskThick=diskthick,
                  intFlux=intflux, phaseCent=phasecen, vOffset=voffset, gasSigma=gassigma, fileName="NGC4234_test").model_cube()

    # Read in data
    hdulist = fits.open('http://www.astro.cardiff.ac.uk/pub/Tim.Davis/NGC4324.fits')
    scidata = hdulist[0].data.T
    scidata = scidata[:, :, :, 0]
    scidata[scidata < np.std(scidata[:, :, 0]) * 4] = 0

    #If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    #script separately:
    KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize, posang=posang, overcube=scidata, pvdthick=4).makeplots(xrange=[-28, 28],
                    yrange=[-28, 28])
    
    return cube


def use_inclouds():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to use the "inclouds" parameter set to create simulations, in this case of a very unrealistic object. Once you
    understand this example then see the "infits" and "inclouds spiral" test for more realistic examples.
    :return: N/A
    """

    # Set up the cube parameters
    xsize = 128
    ysize = 128
    vsize = 1400
    cellsize = 1
    dv = 10
    beamsize = [4, 4, 0]
    inc = 35
    intflux = 30
    posang = 90
    x = np.arange(0, 100, 0.1)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)

    # Define where clouds are in each dimension (x,y,z)
    inclouds = np.array([[40, 0, 0], [39.5075, 6.25738, 0], [38.0423, 12.3607, 0.00000], [35.6403, 18.1596, 0],
                         [32.3607, 23.5114, 0], [28.2843, 28.2843, 0], [23.5114, 32.3607, 0], [18.1596, 35.6403, 0],
                         [12.3607, 38.0423, 0], [6.25737, 39.5075, 0], [0, 40, 0], [-6.25738, 39.5075, 0],
                         [-12.3607, 38.0423, 0], [-18.1596, 35.6403, 0], [-23.5114, 32.3607, 0],
                         [-28.2843, 28.2843, 0], [-32.3607, 23.5114, 0], [-35.6403, 18.1596, 0],
                         [-38.0423, 12.3607, 0], [-39.5075, 6.25738, 0], [-40, 0, 0], [-39.5075, -6.25738, 0],
                         [-38.0423,-12.3607, 0], [-35.6403, -18.1596, 0], [-32.3607, -23.5114, 0], [-28.2843, -28.2843, 0],
                         [-23.5114, -32.3607, 0], [-18.1596, -35.6403, 0], [-12.3607,-38.0423, 0], [-6.25738, -39.5075, 0],
                         [0, -40, 0], [6.25738, -39.5075, 0], [12.3607, -38.0423, 0], [18.1596, -35.6403, 0],
                         [23.5114, -32.3607, 0], [28.2843, -28.2843, 0], [32.3607,-23.5114, 0],  [35.6403, -18.1596, 0],
                         [38.0423, -12.3607, 0], [39.5075, -6.25737, 0], [15, 15, 0], [-15, 15, 0],
                         [-19.8504, -2.44189, 0], [-18.0194, -8.67768, 0], [-14.2856, -13.9972, 0],
                         [-9.04344, -17.8386, 0], [-2.84630, -19.7964, 0], [3.65139, -19.6639, 0],
                         [9.76353, -17.4549, 0], [14.8447, -13.4028, 0], [18.3583, -7.93546, 0],
                         [19.9335, -1.63019, 0]])

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, intFlux=intflux, inClouds=inclouds, velProf=vel,
                 velRad=x, posAng=posang).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize).makeplots()
    
    return cube


def inclouds_spiral():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to use the "inclouds" parameter set to create simulations, in this case of molecular gas in a two armed spiral
    pattern. Any default parameters can be changed by specifying them at the command line (see KinMS.pro or the full
    details of all the available parameters).
    :return: N/A
    """

    # Set up the cube parameters
    xsize = 128 
    ysize = 128 
    vsize = 1400
    cellsize = 1
    dv = 10
    beamsize = 4
    inc = 55
    intflux = 30
    posang = 90

    # Define where clouds are in each dimension (x,y,z) using a logarithmic spiral
    t = np.arange(-20, 20, 0.01)
    a = 0.002 * 60
    b = 0.5
    x1 = a * np.exp(b * t) * np.cos(t)
    y1 = a * np.exp(b * t) * np.sin(t)
    x2 = -a * np.exp(b * t) * np.cos(t)
    y2 = -a * np.exp(b * t) * np.sin(t)

    inclouds = np.empty((len(t) * 2, 3))
    inclouds[:, 0] = np.concatenate((x1, x2)) / 20
    inclouds[:, 1] = np.concatenate((y1, y2)) / 20
    inclouds[:, 2] = np.concatenate((x1 * 0, x2 * 0))

    inclouds = inclouds[(abs(inclouds[:, 0]) > 2) & (abs(inclouds[:, 1]) > 2.0), :]

    # Define a velocity curve
    x = np.arange(0, 5000)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 5000], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, intFlux=intflux, inClouds=inclouds, velProf=vel,
                 velRad=x, posAng=posang).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize, pvdthick=50).makeplots()
    
    return cube


def infits():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to use an input FITS image to create a simulation of what the molecular gas may look like, with a given
    instrument (in this case CARMA). We use a GALEX (Morrissey et al., 2007) FUV image of NGC1437A, and scale it
    assuming the FUV emission comes from star-formation and thus molecular gas, and that the galaxy has a total
    integrated CO flux of 30 Jy km/s. We use the FITS image to set the surface-brightness, and impose a flat velocity
    gradient.
    :return: N/A
    """

    # Set up the cube parameters
    xsize = 128
    ysize = 128
    vsize = 500
    cellsize = 1
    dv = 10
    beamsize = 4
    inc = 0
    posang = 170
    intflux = 30

    # Read in the FITS file and create the "inclouds" variables based on it
    phasecent = [88, 61]

    hdulist = fits.open('http://www.astro.cardiff.ac.uk/pub/Tim.Davis/NGC1437A_FUV.fits')
    fin = hdulist[0].data.T
    s = fin.shape

    xvec = np.arange(0 - phasecent[0], s[0] - phasecent[0]) * (hdulist[0].header['cdelt1'] * 3600)
    yvec = np.arange(0 - phasecent[1], s[1] - phasecent[1]) * (hdulist[0].header['cdelt2'] * 3600)
    w = np.where(fin > 0.002)
    flux_clouds = fin[w]  # Clip the image to avoid FUV noise entering the simulation
    x = xvec[w[0]]
    y = yvec[w[1]]

    inclouds = np.empty((x.size, 3))
    inclouds[:, 0] = x
    inclouds[:, 1] = y
    inclouds[:, 2] = x * 0
    
    ang = np.radians(posang-90)
    velfunc = interpolate.interp1d([-130, 0, 130], [-400, 0, 400], kind='linear')

    vlos_clouds = velfunc(y * np.sin(ang) + x * np.cos(ang))  # Impose a flat velocity profile

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, posang, intFlux=intflux, inClouds=inclouds,
                     vLOS_clouds = vlos_clouds, flux_clouds = flux_clouds).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize).makeplots()
    
    return cube


def veldisp():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to create a simulation of an exponential disk of molecular gas with a velocity dispersion that varies with
    radius.
    :return: N/A
    """

    # Set up cube parameters
    xsize = 128
    ysize = 128
    vsize = 1400
    cellsize = 1
    dv = 10
    beamsize = 2
    intflux = 30
    posang = 90

    # Set up exponential disk SB profile/velocity
    fcent = 10
    scalerad = 20
    inc = 30
    x = np.arange(0, 100, 0.1)
    fx = fcent * np.exp(-x / scalerad)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)
    gassigfunc = interpolate.interp1d([0, 20, 500], [50, 8, 8], kind='linear')
    gassigma = gassigfunc(x)

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel,
                     intFlux=intflux, gasSigma=gassigma, posAng=posang).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize, posang=posang).makeplots(vrange=[-200, 200])
    
    return cube


def diskthick():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to create a simulation of an exponential disk of molecular gas with a thickness that varies with radius. Any
    default parameters can be changed by specifying them at the command line (see KinMS.pro for the full details of all
    the available parameters).
    :return: N/A
    """

    # Set up cube parameters
    xsize = 128
    ysize = 128
    vsize = 1400
    cellsize = 1
    dv = 10
    beamsize = 2
    intflux = 30
    posang = 270

    # Set up exponential disk SB profile/velocity
    fcent = 10
    scalerad = 20
    inc = 90
    x = np.arange(0, 100, 0.1)
    fx = fcent * np.exp(-x / scalerad)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)
    diskthickfunc = interpolate.interp1d([0, 10, 15, 20, 200], [1, 1, 5, 15, 15], kind='linear')
    diskthick = diskthickfunc(x)

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel, intFlux=intflux,
                     diskThick=diskthick, posAng=posang).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize, posang=90).makeplots(xrange=[-30, 30],
    #          yrange=[-30, 30], vrange=[-250, 250])
    
    return cube


def warp():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to create a simulation of a warped exponential disk of molecular gas.
    :return: N/A
    """

    # Set up the cube parameters
    xsize = 128
    ysize = 128
    vsize = 1400
    cellsize = 1
    dv = 10
    beamsize = 2
    intflux = 30

    # Set up exponential disk SB profile/velocity
    fcent = 10
    scalerad = 20
    inc = 60
    x = np.arange(0, 100, 0.1)
    fx = fcent * np.exp(-x / scalerad)
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x)
    diskthickfunc = interpolate.interp1d([0, 15, 50, 500], [270, 270, 300, 300], kind='linear')
    posang = diskthickfunc(x)

    # Create the cube
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc, sbProf=fx, sbRad=x, velProf=vel, intFlux=intflux,
                     posAng=posang).model_cube(toplot=True)

    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize, posang=270).makeplots(vrange=[-250, 250])
    
    return cube
    

def retclouds():
    """
    A test procedure to demonstrate the KinMS code, and check if it works on your system. This procedure demonstrates
    how to use the return clouds feature to recursivly build models - here a misaligned central and outer disc.
    :return: N/A
    """

    # Set up the cube parameters
    xsize = 64
    ysize = 64
    vsize = 1000
    cellsize = 1
    dv = 10
    beamsize = [4, 4, 0]
    intflux = 30
    posang = 90
    gassigma = 10

    # Set up exponential disk SB profile/velocity for disc one
    inc1 = 75
    x1 = np.arange(0, 100, 0.1)
    fx1 = np.exp(-x1 / 4)
    fx1[x1 > 5] = 0
    velfunc = interpolate.interp1d([0, 0.5, 1, 3, 500], [0, 50, 100, 210, 210], kind='linear')
    vel = velfunc(x1)
    nsamps1 = int(5e4)

    # Run KinMS for disc 1
    _, inclouds1, vlos1 = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc1, posang+180, sbProf=fx1, sbRad=x1, velProf=vel,
                        nSamps=nsamps1, intFlux=intflux, gasSigma=gassigma, returnClouds=True).model_cube()

    # Set up exponential disk SB profile for disc two
    inc2 = 35
    x2 = np.arange(0, 100, 0.1)
    fx2 = np.exp(-x2 / 15)
    fx2[x2 < 10] = 0
    nsamps2 = int(1e6)

    # Run KinMS for disc 2
    _, inclouds2, vlos2 = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc2, posang, sbProf=fx2, sbRad=x2, velProf=vel,
                        nSamps=nsamps2, intFlux=intflux, gasSigma=gassigma, returnClouds=True).model_cube()

    # Combine
    inclouds = np.concatenate((inclouds1, inclouds2), axis=0)
    vlos = np.concatenate((vlos1, vlos2))

    # Make a cube for the whole thing
    cube = KinMS(xsize, ysize, vsize, cellsize, dv, beamsize, inc1, posang, inClouds=inclouds, vLOS_clouds=vlos,
                     intFlux=intflux).model_cube(toplot=True)
    
    # If you want to change something about the plots, or save them directly to your disk, you can use the plotting
    # script separately:
    #KinMS_plotter(cube, xsize, ysize, vsize, cellsize, dv, beamsize, posang=posang).makeplots()
    
    return cube


def run_tests():
    have_connection=internet()
    if have_connection: 
        print("Test - simulate the gas ring in NGC4324")
        print("[Close plot to continue]")
        cube = ngc4324()
    print("Test - simulate an exponential disk")
    print("[Close plot to continue]")
    cube = expdisk()
    print("Test - using the INCLOUDS mechanism - unrealistic")
    print("[Close plot to continue]")
    cube = use_inclouds()
    print("Test - using the INCLOUDS mechanism - realistic")
    print("[Close plot to continue]")
    cube = inclouds_spiral()
    if have_connection: 
        print("Test - using a FITS file as input")
        print("[Close plot to continue]")
        cube = infits()
    print("Test - using variable velocity dispersion")
    print("[Close plot to continue]")
    cube = veldisp()
    print("Test - using variable disk thickness")
    print("[Close plot to continue]")
    cube = diskthick()
    print("Test - simulate a warped exponential disk")
    print("[Close plot to finish]")
    cube = warp()
    print("Test - using the returnclouds mechanism")
    print("[Close plot to finish]")
    cube = retclouds()
    print("Test - using the gravgas mechanism")
    cube1, cube2 = expdisk_gasgrav()
    del(cube); del(cube1); del(cube2)

#=============================================================================#
#/// END OF SCRIPT ///////////////////////////////////////////////////////////#
#=============================================================================#
    