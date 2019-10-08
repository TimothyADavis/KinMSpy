#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:44:09 2019

@author: jamesdawson
"""



"""
MAKEBEAM

:param xpixels: Number of pixels in the x-axis
:param ypixels: Number of pixels in the y-axis
:param beamSize: Beam information. This can be a list/numpy array of length 2 or 3: the first two elements
                 contain the sizes of the major and minor axes (the order does not matter) in arcseconds (RIGHT?? -Yes in arcseconds),
                 the third element is the position angle. If no third element is given, the angle is assumed to
                 be zero. If an integer/float is given, a circular beam of that size is assumed.
:param cellSize:
:param cent:
:return:
"""


"""

kinms_sampleFromArbDist_oneSided

This function takes the input radial distribution and generates the positions of
"nSamps" cloudlets from under it. It also accounts for disk thickness if requested.

Parameters
----------
sbRad : np.ndarray or list of floats
        Radius vector (in units of pixels).

sbProf : np.ndarray or list of floats
        Surface brightness profile (arbitrarily scaled).

nSamps : int or float
        Number of samples to draw from the distribution (default is 5e5).

fixSeed : bool
        Whether to use a fixed (or random) seed (list of four integers).

diskThick : float or np.ndarray/list of floats
     (Default value = 0.0)
     The disc scaleheight. If a single value then this is used at all radii.
     If a list/ndarray then it should have the same length as sbRad, and will be
     the disc thickness as a function of sbRad.

Returns
-------
inClouds : np.ndarray of floats
        Returns an ndarray of "nSamps" by 3 in size. Each row corresponds to the x, y, z position of a cloudlet.
"""


"""

kinms_create_velField_oneSided

This function takes the input circular velocity distribution
and the position of point sources and creates the velocity field
taking into account warps, inflow/outflow etc as required.

Parameters
----------
velRad : np.ndarray of double
        Radius vector (in units of pixels).

velProf : np.ndarray of double
        Velocity profile (in units of km/s).

r_flat : np.ndarray of double
        Radius of each cloudlet from the kinematic centre
        in the plane of the disc. Units of pixels.

inc : double or np.ndarray of double
        Inclination of the disc, using the usual astronomical convention.
        Can be either a double, or an array of doubles. If single valued
        then the disc is flat. If an array is passed then it should
        describe how the galaxy inclination changes as a function of `velrad`.
        Used to create inclination warps.

posAng : double or np.ndarray of double
        Position angle of the disc, using the usual astronomical convention.
        Can be either a double, or an array of doubles. If single valued
        then the disc major axis is straight. If an array is passed then it should
        describe how the position angle changes as a function of `velrad`.
        Used to create position angle warps.

gasSigma : double or np.ndarray of double
        Velocity dispersion of the gas. Units of km/s.
        Can be either a double, or an array of doubles. If single valued
        then the velocity dispersion is constant throughout the disc.
        If an array is passed then it should describe how the velocity
        dispersion changes as a function of `velrad`.

fixSeed : list of int
        List of length 4 containing the seeds for random number generation.

xPos : np.ndarray of double
        X position of each cloudlet. Units of pixels.
        
Pos : np.ndarray of double
        Y position of each cloudlet. Units of pixels.

vPhaseCent : list of double
     (Default value = [0, 0])
        Kinematic centre of the rotation in the x-y plane. Units of pixels.
        Used if the kinematic and morphological centres are not the same.

vPosAng : double or np.ndarray of double
     (Default value = False)
        Kinematic position angle of the disc, using the usual astronomical convention.
        Can be either a double, or an array of doubles. If single valued
        then the disc kinematic major axis is straight. If an array is passed then it should
        describe how the kinematic position angle changes as a function of `velrad`.
        Used if the kinematic and morphological position angles are not the same.

vRadial : double or np.ndarray of double
     (Default value = 0)
        Magnitude of inflow/outflowing motions (km/s). Negative
        numbers here are inflow, positive numbers denote
        outflow. These are included in the velocity field using
        formalism of KINEMETRY (Krajnović et al. 2006 MNRAS, 366, 787).
        Can input a constant or a vector, giving the radial
        motion as a function of the radius vector
        `velrad`. Default is no inflow/outflow.

posAng_rad : double or np.ndarray of double
     (Default value = 0)
        Position angle of the disc at the position `r_flat` of each cloudlet.

inc_rad : double or np.ndarray of double
     (Default value = 0)
        Inclination angle of the disc at the position `r_flat` of each cloudlet.

Returns
-------
los_vel : np.ndarray of double
        Line of sight velocity of each cloudlet, in km/s.

"""


"""

gasGravity_velocity

This function takes the position of the input cloudlets, and calculates the
potential, and thus the increase in the circular velocity due to the gas mass itself.

Parameters
----------
xPos : np.ndarray of double
        X position of each cloudlet. Units of arcseconds.
 
yPos : np.ndarray of double
        Y position of each cloudlet. Units of arcseconds.

zPos : np.ndarray of double
        Z position of each cloudlet. Units of arcseconds

massDist : list of double
        List of [gasmass,distance] - gas mass in solar masses, distance in Mpc.

velRad : np.ndarray of double
        Radius vector (in units of pixels).

Returns
-------
np.ndarray of double
        Addition to the circular velocity just due to the mass of the gas itself, in units of km/s.
"""


"""

model_cube

The main KinMS function. Takes inputs specifing the observing parameters and type of model.
Returns the created model cube.

Parameters
----------
xs : float
    X-axis size for resultant cube (in arcseconds)

ys : float
    Y-axis size for resultant cube (in arcseconds)

vs : float
    Velocity axis size for resultant cube (in km/s)

cellsize : float
    Pixel size required (arcsec/pixel)

dv : float
    Channel size in velocity direction (km/s/channel)

beamsize : float or list of float
    Scalar or three element list for size of convolving
    beam (in arcseconds).  If a scalar then beam is assumed
    to be circular. If a vector then denotes beam major
    axis size in element zero, and the beam minor axis in
    element one. The beam position angle should be given in
    element two. I.e. [bmaj,bmin,bpa].

inc :   double or np.ndarray of double
    Inclination angle of the gas disc on the sky
    (degrees). Can input a constant or a vector,
    giving the inclination as a function of the
    radius vector `velrad` (in order to model warps etc)

gassigma : double or np.ndarray of double, optional
     (Default value = 0)
    Velocity dispersion of the gas. Units of km/s.
    Can be either a double, or an array of doubles. If single valued
    then the velocity dispersion is constant throughout the disc.
    If an array is passed then it should describe how the velocity
    dispersion changes as a function of `velrad`.

sbprof : np.ndarray of double, optional
     (Default value = [])
    Surface brightness profile (arbitrarily scaled) as a function of `sbrad`.

sbrad : np.ndarray of double, optional
     (Default value = [])
    Radius vector for surface brightness profile (units of arcseconds).

velrad : np.ndarray of double, optional
     (Default value = [])
    Radius vector for velocity profile (units of arcseconds).

velprof : np.ndarray of double, optional
     (Default value = [])
    Circular velocity profile (in km/s) as a function of `velrad`.

diskthick : double or np.ndarray of double, optional
     (Default value = 0)
    The disc scaleheight in arcseconds. If a single value then this is used at all radii.
    If a ndarray then it should have the same length as `sbrad`, and will be
    the disc thickness as a function of `sbrad`.

cleanout : bool, optional
     (Default value = False)
    If set then do not convolve with the beam, and output the
    "clean components". Useful to create input for other
    simulation tools (e.g sim_observe in CASA).

nsamps : int, optional
     (Default value = 100000)
    Number of cloudlets to use to create the model. Large numbers
    will reduce numerical noise (especially in large cubes),
    at the cost of increasing runtime.

posang : double or np.ndarray of double, optional
     (Default value = 0.0)
    Position angle of the disc, using the usual astronomical convention.
    Can be either a double, or an array of doubles. If single valued
    then the disc major axis is straight. If an array is passed then it should
    describe how the position angle changes as a function of `velrad`.
    Used to create position angle warps.

intflux : double, optional
     (Default value = 0)
    Total integrated flux you want the output gas to
    have. (In Jy/km/s).

inclouds : np.ndarray of double, optional
     (Default value = [])
    If your required gas distribution is not symmetric you
    may input vectors containing the position of the
    clouds you wish to simulate. This 3-vector should
    contain the X, Y and Z positions, in units of arcseconds
    from the phase centre. If this variable is used, then
    `diskthick`, `sbrad` and `sbprof` are ignored.
    Example: INCLOUDS=[[0,0,0],[10,-10,2],...,[xpos,ypos,zpos]]

vlos_clouds : np.ndarray of double, optional
     (Default value = [])
    This vector should contain the LOS velocity for
    each point defined in INCLOUDS, in units of km/s. If
    not supplied then INCLOUDS is assumed to be the -face
    on- distribution and that VELPROF/VELRAD should be
    used, and the distribution projected. If this
    variable is used then GASSIGMA/INC are ignored.

flux_clouds : np.ndarray of double, optional
     (Default value = 0)
    This vector can be used to supply the flux of each
    point in INCLOUDS. If used alone then total flux in the model is equal
    to total(FLUX_INCLOUDS). If INTFLUX used then this vector denotes
    the relative brightness of the points in
    INCLOUDS.

phasecen : np.ndarray of double, optional
     (Default value = np.array([0., 0.])
    This two dimensional array specifies the morphological centre of the
    disc structure you create with respect to the central pixel of the
    generated cube.

returnclouds: bool, optional
    (Default value= False)
    If set True then KinMS returns the created `inclouds` and `vlos_clouds`
    in addition to the cube.

Other Parameters
----------------

filename : string or bool, optional
     (Default value = False)
    If you wish to save the resulting model to a fits file, set this variable.
    The output filename will be `filename`_simcube.fits

ra : float, optional
     (Default value = 0)
    RA to use in the header of the output cube (in degrees).

dec : float, optional
     (Default value = 0)
    DEC to use in the header of the output cube (in degrees).

restfreq : double, optional
     (Default value = 115.271e9)
    Rest-frequency of spectral line of choice (in Hz). Only
    matters if you are outputting a FITS file  Default: 12CO(1-0)

vsys : double, optional
     (Default value = 0)
    Systemic velocity (km/s).

Returns
-------

cube : np.ndarray of double
    Returns the created cube as a 3 dimensional array

inclouds: np.ndarray of double
    If `returnclouds` is set then this is returned, containing
    the cloudlets generated by KinMS

vlos_clouds: np.ndarray of double
    If `returnclouds` is set then this is returned, containing
    the LOS velocities of cloudlets generated by KinMS

"""