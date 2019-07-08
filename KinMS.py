# coding: utf-8
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

import numpy as np
import scipy.integrate
from scipy import interpolate
from astropy.io import fits
from astropy.convolution import convolve_fft

import datetime # FOR TESTING, REMOVE LATER

class KinMS:

    def __init__(self):

        self.diskThick = 0
        self.nSamps = 5e5
        self.fixedSeed = np.array([100, 101, 102, 103])
        self.randomSeed = np.random.randint(0, 100, 4)
        self.vRadial = 0
        self.vPhaseCent = np.zeros(2)
        self.posAng_rad = 0
        self.inc_rad = 0
        self.gasSigma = 0
        self.ra = 0
        self.dec = 0
        self.posAng = 0
        self.intFlux = 0
        self.flux_clouds = 0
        self.vSys = 0
        self.phaseCent = np.zeros(2)
        self.vOffset = 0
        self.restFreq = 115.271e9
        self.vPosAng = 0
        self.sbProf = []
        self.sbRad = []
        self.velRad = []
        self.velProf = []
        self.inClouds = []
        self.vLOS_clouds = []

    def print_variables(self, param_dict):

        # Make sure variables are printed in the right order and split up into user defined, default, and bools.
        keys = list(param_dict.keys())[::-1]
        values = list(param_dict.values())[::-1]
        default_keys = []
        default_values = []
        bool_keys = []
        bool_values = []

        print("\n\n*** Hello and welcome to the grand KinMSpy :D ***")

        print('_' * 37 + '\n \n' + 'Setting user defined variables to: \n')

        for i in range(len(keys)):
            if values[i][1] == 0:
                print(keys[i] + ' = ' + str(values[i][0]))
            elif values[i][1] == 1:
                default_keys.append(keys[i])
                default_values.append(values[i])
            else:
                bool_keys.append(keys[i])
                bool_values.append(values[i])

        print('\nSetting default values to: \n')

        for i in range(len(default_keys)):
            print(default_keys[i] + ' = ' + str(default_values[i][0]))

        print('\nSetting options to: \n')

        for i in range(len(bool_keys)):
            print(bool_keys[i] + ' = ' + str(bool_values[i][0]))

        print('_' * 37)

        return

    def makebeam(self, xpixels, ypixels, beamSize, cellSize=1, cent=None):
        """
        :param xpixels: Number of pixels in the x-axis
        :param ypixels: Number of pixels in the y-axis
        :param beamSize: Beam information. This can be a list/numpy array of length 2 or 3: the first two elements
                         contain the sizes of the major and minor axes (the order does not matter) in arcseconds (RIGHT??),
                         the third element is the position angle. If no third element is given, the angle is assumed to
                         be zero. If an integer/float is given, a circular beam of that size is assumed.
        :param cellSize:
        :param cent:
        :return:
        """

        if not cent: cent = [xpixels / 2, ypixels / 2]

        beamSize = np.array(beamSize)

        try:
            if len(beamSize) == 2:
                beamSize = np.append(beamSize, 0)
            if beamSize[1] > beamSize[0]:
                beamSize[1], beamSize[0] = beamSize[0], beamSize[1]
        except:
            beamSize = np.array([beamSize, beamSize, 0])

        st_dev = beamSize[0:2] / cellSize / 2.355

        rot = beamSize[2]

        if np.tan(np.radians(rot)) == 0:
            dirfac = 1
        else:
            dirfac = np.sign(np.tan(np.radians(rot)))

        x, y = np.indices((int(xpixels), int(ypixels)), dtype='float')

        x -= cent[0]
        y -= cent[1]

        a = (np.cos(np.radians(rot)) ** 2) / (2 * st_dev[1] ** 2) + (np.sin(np.radians(rot)) ** 2) / \
            (2 * (st_dev[0] ** 2))

        b = (dirfac * (np.sin(2 * np.radians(rot)) ** 2) / (4 * st_dev[1] ** 2)) + ((-1 * dirfac) * \
            (np.sin(2 * np.radians(rot)) ** 2) / (4 * st_dev[0] ** 2))

        c = (np.sin(np.radians(rot)) ** 2) / (2 * st_dev[1] ** 2) + (np.cos(np.radians(rot)) ** 2) / \
            (2 * st_dev[0] ** 2)

        psf = np.exp(-1 * (a * x ** 2 - 2 * b * (x * y) + c * y ** 2))

        return psf

    def kinms_sampleFromArbDist_oneSided(self, sbRad, sbProf, fixSeed, nSamps, diskThick):
        """

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

        if self.verbose: print('Generating cloudlets,', end=' ')

        # If variables are not entered by user, adopt default (global) values.
        if fixSeed:
            seed = self.fixedSeed
        else:
            seed = self.randomSeed

        # Set everything to numpy arrays to accept list input
        sbRad = np.array(sbRad)
        sbProf = np.array(sbProf)

        nSamps = int(nSamps)

        # Randomly generate the radii of clouds based on the distribution given by the brightness profile.
        px = scipy.integrate.cumtrapz(sbProf * 2 * np.pi * abs(sbRad), abs(sbRad), initial=0)
        px /= max(px)
        rng1 = np.random.RandomState(seed[0])
        pick = rng1.random_sample(nSamps)
        interpfunc = interpolate.interp1d(px, sbRad, kind='linear')
        r_flat = interpfunc(pick)

        # Generates a random phase around the galaxy's axis for each cloud.
        rng2 = np.random.RandomState(seed[1])
        phi = rng2.random_sample(nSamps) * 2 * np.pi

        # Find the thickness of the disk at the radius of each cloud.
        try:
            if len(diskThick) != len(sbRad):
                print('\n \n ... Please make sure the length of diskThick is the same as that of sbRad! Returning.')
                return

            elif len(diskThick) > 0:
                diskThick = np.array(diskThick)
                interpfunc2 = interpolate.interp1d(sbRad, diskThick, kind='linear')
                diskThick_here = interpfunc2(r_flat)
                if self.verbose: print('using the scale height profile provided.')
            else:
                diskThick_here = diskThick
                if self.verbose: print('using a constant scale height of ' + str(diskThick) + '.')

        except:
            diskThick_here = diskThick
            if self.verbose: print('using a constant scale height of ' + str(diskThick) + '.')

        # Generates a random (uniform) z-position satisfying |z|<disk_here.
        rng3 = np.random.RandomState(seed[3])
        zPos = diskThick_here * rng3.uniform(-1, 1, nSamps)

        # Calculate the x & y position of the clouds in the x-y plane of the disk.
        r_3d = np.sqrt((r_flat ** 2) + (zPos ** 2))
        theta = np.arccos(zPos / r_3d)
        xPos = ((r_3d * np.cos(phi) * np.sin(theta)))
        yPos = ((r_3d * np.sin(phi) * np.sin(theta)))

        # Generates the output array
        inClouds = np.vstack((xPos, yPos, zPos)).T

        return inClouds
    
    def kinms_create_velField_oneSided(self, velRad, velProf, r_flat, inc, posAng, gasSigma, xPos, yPos, fixSeed,
                                       vPhaseCent, vRadial, posAng_rad, inc_rad, vPosAng):
            

        """
        
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
        
        ### MAKE EVERYTHING AN ARRAY IN HERE RATHER THAN A LIST OR DOUBLE ###
        
        if fixSeed:
            seed = self.fixedSeed
        else:
            seed = self.randomSeed

        parameter_dictionary = {}
        parameter_dictionary['vRadial'] = vRadial
        parameter_dictionary['vPhaseCent'] = vPhaseCent
        parameter_dictionary['posAng_rad'] = posAng_rad
        parameter_dictionary['inc_rad'] = inc_rad  
        #self.print_variables(parameter_dictionary)
                  
        velInterFunc = interpolate.interp1d(velRad,velProf,kind='linear')
        vRad = velInterFunc(r_flat)
        # Calculate a peculiar velocity for each cloudlet based on the velocity dispersion
        rng4 = np.random.RandomState(seed[3]) 
        velDisp = rng4.randn(len(xPos))
        try:
                if len(gasSigma) > 0:
                    gasSigmaInterFunc = interpolate.interp1d(velRad,gasSigma,kind='linear')
                    velDisp *= gasSigmaInterFunc(r_flat)
                else:
                    velDisp *= gasSigma
        except:
                velDisp *= gasSigma
                
        # Find the rotation angle so the velocity field has the correct position angle (allows warps)
        if not vPosAng:
            ang2rot=0
        else:
            try:
                if len(vPosAng) > 0:
                    vPosAngInterFunc = interpolate.interp1d(velRad,vPosAng,kind='linear')
                    vPosAng_rad = vPosAngInterFunc(r_flat)
                else:
                    vPosAng_rad = np.full(len(r_flat), vPosAng, np.double)
                    ang2rot = ((posAng_rad - vPosAng_rad))
            except:
                vPosAng_rad = np.full(len(r_flat),vPosAng,np.double)
                ang2rot = ((posAng_rad-vPosAng_rad))
        #Calculate the los velocity for each cloudlet
        los_vel = velDisp                                                                                                                    
        los_vel += (-1) * vRad * (np.cos(np.arctan2((yPos + vPhaseCent[1]),(xPos + vPhaseCent[0])) + (np.radians(ang2rot))) * np.sin(np.radians(inc_rad)))
        #Add radial inflow/outflow
        try:
            if len(vRadial) > 0:
                vRadialInterFunc = interpolate.interp1d(velRad,vRadial,kind='linear')
                vRadial_rad = vRadialInterFunc(r_flat)
            else:
                vRadial_rad = np.full(len(r_flat), vRadial, np.double)
        except:
            vRadial_rad=np.full(len(r_flat),vRadial,np.double)
        los_vel += vRadial_rad * (np.sin(np.arctan2((yPos+vPhaseCent[1]),(xPos + vPhaseCent[0])) + (np.radians(ang2rot))) * np.sin(np.radians(inc_rad)))
        # Output the array of los velocities
        return los_vel

    def save_fits(self, fileName, cube, cellSize, dv, cent, ra, dec, vSys, beamSize):

        hdu = fits.PrimaryHDU(cube.T)

        hdu.header['CDELT1'] = cellSize / -3600
        hdu.header['CDELT2'] = cellSize / 3600
        hdu.header['CDELT3'] = dv * 1000
        hdu.header['CRPIX1'] = cent[0] - 1
        hdu.header['CRPIX2'] = cent[1] - 1
        hdu.header['CRPIX3'] = cent[2]
        hdu.header['CRVAL1'] = ra
        hdu.header['CRVAL2'] = dec
        hdu.header['CRVAL3'] = vSys * 1000, 'm/s'
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        hdu.header['CUNIT3'] = 'm/s'
        hdu.header['BSCALE'] = 1
        hdu.header['BZERO'] = 0
        hdu.header['BMIN'] = np.min(np.array(beamSize[0:2]) / 3600)
        hdu.header['BMAJ'] = np.max(np.array(beamSize[0:2]) / 3600)
        hdu.header['BTYPE'] = 'Intensity'
        hdu.header['BPA'] = beamSize[2]
        hdu.header['CTYPE1'] = 'RA---SIN'
        hdu.header['CTYPE2'] = 'DEC--SIN'
        hdu.header['CTYPE3'] = 'VRAD'
        hdu.header['EQUINOX'] = 2000
        hdu.header['RADESYS'] = 'FK5'
        hdu.header['BUNIT'] = 'Jy/beam'
        hdu.header['SPECSYS'] = 'BARYCENT'

        hdu.writeto(fileName + '_simcube.fits', overwrite=True, output_verify='fix')

        return
      
    def gasGravity_velocity(self, xPos, yPos, zPos, massDist, velRad):
        """
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
        
        xPos = np.array(xPos); yPos = np.array(yPos); zPos = np.array(zPos);
        massDist = np.array(massDist); velRad = np.array(velRad)
        
        rad = np.sqrt( (xPos**2) + (yPos**2) + (zPos**2) )						                ## 3D radius
        cumMass = ((np.arange(xPos.size + 1)) * (massDist[0] / xPos.size))					    ## cumulative mass

        
        #max_rad = np.argmax(rad)
        #max_velRad =  velRad[max_rad]+1
        #print(max_velRad)
        
        #print(np.max(velRad).clip(1,max=None))
        #print(np.max(rad))
        
        max_velRad = np.max(velRad).clip(min=np.max(rad), max=None)+1
        #print(max_velRad)
        
        print(rad)
        new_rad = np.insert(sorted(rad),0,0)
        print(new_rad)
        
        ptcl_rad = np.append(new_rad, max_velRad)
        cumMass_max_end = np.append(cumMass,np.max(cumMass))

        cumMass_interFunc = interpolate.interp1d(ptcl_rad,cumMass_max_end,kind='linear')
        if velRad[0] == 0.0:
            return 	np.append(0.0,np.sqrt((4.301e-3 * cumMass_interFunc(velRad[1:]))/(4.84 * velRad[1:] * massDist[1])))    ## return velocity
        else:
            return 	np.sqrt((4.301e-3 * cumMass_interFunc(velRad))/(4.84 * velRad * massDist[1]))
          
    def model_cube(self, xs, ys, vs, cellSize, dv, beamSize, inc, gasSigma=None, diskThick=None, ra=None, dec=None,
           nSamps=None, posAng=None, intFlux=None, flux_clouds=None, vSys=None, phaseCent=None, vOffset=None, \
           vRadial=None, vPosAng=None, vPhaseCent=None, restFreq=None, sbProf=None, sbRad=None, velRad=None,
           velProf=None, inClouds=None, vLOS_clouds=None, fileName=False, fixSeed=False, cleanOut=False,
           returnClouds=False, gasGrav=False, verbose=False):
          
        """

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

        # Set all values that were not defined by user to default values and make sure the right values get printed
        local_vars = locals()
        global_vars = vars(self)
        print_dict = {} # 0 = user defined, 1 = default, 2 = bool

        for k, v in local_vars.items():
            try:
                if not v == None:
                    global_vars[k] = local_vars[k]
                    if k != 'self':
                        if type(v) != type(True):
                            print_dict[k] = (global_vars[k], 0)
                        else:
                            print_dict[k] = (global_vars[k], 2)
                else:
                    print_dict[k] = (global_vars[k], 1)
            except:
                global_vars[k] = local_vars[k]
                print_dict[k] = ('User defined array of length ' + str(len(global_vars[k])), 0)

        self.__dict__.update(global_vars)

        if verbose:
            self.verbose = True
            self.print_variables(print_dict)

        # Set variables to numpy arrays if necessary
        self.velProf = np.array(self.velProf)
        self.velRad = np.array(self.velRad)

        # Work out images sizes
        xSize = np.round(xs / cellSize)
        ySize = np.round(ys / cellSize)
        vSize = np.round(vs / dv)

        cent = [(xSize / 2) + (self.phaseCent[0] / cellSize), (ySize / 2) + (self.phaseCent[1] / cellSize),
                (vSize / 2) + (self.vOffset / dv)]

        vPhaseCent = self.vPhaseCent / [cellSize, cellSize]

        # If cloudlets not previously specified, generate them
        if not len(self.inClouds):
            if not len(self.sbRad) or not len(self.sbProf):
                print('\nPlease define either \"inClouds\" or \"sbRad\" and \"sbProf\". Returning.')
                return
            else:
                self.inClouds = self.kinms_sampleFromArbDist_oneSided(self.sbRad, self.sbProf, fixSeed,
                                                                      self.nSamps, self.diskThick)

        xPos = (self.inClouds[:, 0] / cellSize)
        yPos = (self.inClouds[:, 1] / cellSize)
        zPos = (self.inClouds[:, 2] / cellSize)
        r_flat = np.sqrt((xPos * xPos) + (yPos * yPos))

        # Find the los velocity and cube position of the clouds
        # If los velocity specified, assume that the clouds have already been projected correctly.
        if len(self.vLOS_clouds):
            los_vel = self.vLOS_clouds
            x2 = xPos
            y2 = yPos
            z2 = zPos

        # If los velocities not specified, calculate them.
        # Include the potential of the gas.
        elif not velRad or not velProf:
            print('\nPlease define either \"vLOS_clouds\" or \"velRad\" and \"velProf\". Returning.')
            return

        else:
            if not gasGrav:
                # --> This function is a mess so check this line after it is fixed! <--
                #gasGravVel = self.gasGravity_velocity(xPos * cellSize, yPos * cellSize, zPos * cellSize, gasGrav, velRad)
                gasGravVel = 1  # Dummy
                velProf = np.sqrt((self.velProf ** 2) + (gasGravVel ** 2))

            self.posAng = 90 - self.posAng

            try:
                if len(self.posAng) > 0:
                    posAngRadInterFunc = interpolate.interp1d(self.velRad, self.posAng, kind='linear')
                    posAng_rad = posAngRadInterFunc(r_flat * cellSize)
                else:
                    posAng_rad = np.full(len(r_flat), self.posAng, float)
            except:
                posAng_rad = np.full(len(r_flat), self.posAng, float)

            try:
                if len(inc) > 0:
                    incRadInterFunc = interpolate.interp1d(self.velRad, inc, kind='linear')
                    inc_rad = incRadInterFunc(r_flat * cellSize)
                else:
                    inc_rad = np.full(len(r_flat), inc, float)
            except:
                inc_rad = np.full(len(r_flat), inc, float)

            # Calculate the LOS velocity.
            los_vel = self.kinms_create_velField_oneSided((self.velRad / cellSize), self.velProf, r_flat, inc, \
                      self.posAng, self.gasSigma, xPos, yPos, fixSeed=fixSeed, vPhaseCent=self.vPhaseCent, \
                      vRadial = self.vRadial, posAng_rad=self.posAng_rad, inc_rad=inc_rad, vPosAng=self.vPosAng)

            # Project the clouds to take into account inclination.
            c = np.cos(np.radians(inc_rad))
            s = np.sin(np.radians(inc_rad))
            x2 = xPos
            y2 = (c * yPos) + (s * zPos)
            z2 = (-s * yPos) + (c * zPos)

            # Correct orientation by rotating by position angle.
            ang = self.posAng_rad
            c = np.cos(np.radians(ang))
            s = np.sin(np.radians(ang))
            x3 = (c * x2) + (s * y2)
            y3 = (-s * x2) + (c * y2)
            x2 = x3
            y2 = y3

        # Now add the flux into the cube.
        # Centre the clouds in the cube on the centre of the object.
        los_vel_dv_cent2 = np.round((los_vel / dv) + cent[2])
        x2_cent0 = np.round(x2 + cent[0])
        y2_cent1 = np.round(y2 + cent[1])

        # Find the reduced set of clouds that lie inside the cube.
        subs = np.where(((x2_cent0 >= 0) & (x2_cent0 < xSize) & (y2_cent1 >= 0) & (y2_cent1 < ySize) & \
                         (los_vel_dv_cent2 >= 0) & (los_vel_dv_cent2 < vSize)))[0]

        clouds2do = np.vstack((x2_cent0[subs], y2_cent1[subs], los_vel_dv_cent2[subs])).T


        ### START HERE ###


        # If there are clouds to use, and we know the flux of each cloud, add them to the cube.
        # If not, bin each position to get a relative flux.
        if len(subs) > 0:
            if not isinstance(self.flux_clouds, (list, tuple, np.ndarray)):
                cube, edges = np.histogramdd(clouds2do, bins=(xSize, ySize, vSize),
                                             range=((0, xSize), (0, ySize), (0, vSize)))
            else:
                cube = np.zeros((np.int(xSize), np.int(ySize), np.int(vSize)))
                flux_clouds = self.flux_clouds[subs]
                for i in range(0, len(subs)):
                    const = flux_clouds[i]
                    csub = (int(clouds2do[i, 0]), int(clouds2do[i, 1]), int(clouds2do[i, 2]))
                    cube[csub] = cube[csub] + const
        else:
            cube = np.zeros((np.int(xSize), np.int(ySize), np.int(vSize)))

        return

        # Convolve with the beam point spread function to obtain a dirty cube
        if not cleanOut:
            psf = self.makebeam(xSize, ySize, beamSize)
            w2do = np.where(cube.sum(axis=0).sum(axis=0) > 0)[0]
            for i in range(0, w2do.size): cube[:, :, w2do[i]] = convolve_fft(cube[:, :, w2do[i]], psf)
        # Normalise by the known integrated flux
        if intFlux > 0:
            if not cleanOut:
                cube *= ((intFlux * psf.sum()) / (cube.sum() * dv))
            else:
                cube *= ((intFlux) / (cube.sum() * dv))
        else:
            if isinstance(flux_clouds, (list, tuple, np.ndarray)):
                cube *= (flux_clouds.sum() / cube.sum())
            else:
                cube /= cube.sum()

        # If appropriate, generate the FITS file header and save to disc
        if fileName:
            self.save_fits(fileName, cube, cellSize, dv, cent, ra, dec, vSys, beamSize)

        # Output the final cube
        if returnClouds:
            retClouds = np.empty((nSamps, 3))
            retClouds[:, 0] = x2 * cellSize
            retClouds[:, 1] = y2 * cellSize
            retClouds[:, 2] = z2 * cellSize
            return cube, retClouds, los_vel
        else:
            return cube

          
#### TESTY TEST ####
          
KinMS().model_cube(30, 30, 30, 2, 10, 3, 76, nSamps=100, verbose=False, sbProf=[1,2,3], sbRad = [1,2,3], diskThick=10,
                   velRad=[1,2,3], velProf=[1,2,3])

#KinMS().kinms_create_velField_oneSided(velRad=np.array([0,1,2]),velProf=np.array([1,1,1]),r_flat=np.array([0,1,2]),
#      inc=90,posAng=45,gasSigma=np.array([1,1,1]),xPos=np.array([0,1,2]),yPos=np.array([0,1,2]))

#KinMS().kinms_create_velField_oneSided(velRad=np.array([0,1,2]),velProf=np.array([1,1,1]),r_flat=np.array([0,1,2]),
#      inc=90,posAng=45,gasSigma=np.array([1,1,1]),xPos=np.array([0,1,2]),yPos=np.array([0,1,2]))

#KinMS().gasGravity_velocity([1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,2,3])