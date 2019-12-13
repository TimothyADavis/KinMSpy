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

#=============================================================================#
#/// IMPORT PACKAGES /////////////////////////////////////////////////////////#
#=============================================================================#

import numpy as np
import scipy.integrate
from scipy import interpolate
from astropy.io import fits
from astropy.convolution import convolve_fft
import warnings; warnings.filterwarnings("ignore")
from KinMS_figures import KinMS_plotter

#=============================================================================#
#/// START OF CLASS //////////////////////////////////////////////////////////#
#=============================================================================#

class KinMS:
        
    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def __init__(self, xs, ys, vs, cellSize, dv, beamSize, inc, posAng, gasSigma=0, diskThick=0, flux_clouds=0, 
                 sbProf=[], sbRad=[], velRad=[], velProf=[], inClouds=[], vLOS_clouds=[], massDist=[], vRadial=[],
                 ra=None, dec=None, nSamps=None, seed=None, intFlux=None, vSys=None, phaseCent=None, vOffset=None,
                 vPosAng=None, vPhaseCent=None, restFreq=None, fileName=False, fixSeed=False,
                 cleanOut=False, returnClouds=False, verbose=False, toplot=False):

        self.xs = xs
        self.ys = ys
        self.vs = vs
        self.cellSize = cellSize
        self.dv = dv
        self.beamSize = beamSize
        self.inClouds = np.array(inClouds); self.inClouds_given = True
        self.vLOS_clouds = np.array(vLOS_clouds) 
        self.massDist = np.array(massDist)
        self.ra = ra 
        self.dec = dec
        self.seed = seed or np.array([100, 101, 102, 103], dtype='int')
        self.intFlux = intFlux or 0
        self.vSys = vSys
        self.phaseCent = phaseCent or np.zeros(2)
        self.vOffset = vOffset or 0
        self.vPosAng = vPosAng or 0
        self.vPhaseCent = vPhaseCent or np.zeros(2)
        self.restFreq = restFreq or 115.271e9
        self.fileName = fileName
        self.fixSeed = fixSeed
        self.cleanOut = cleanOut
        self.returnClouds = returnClouds
        self.verbose = verbose
        self.toplot = toplot

        if not nSamps:
            self.nSamps = int(5e5)
        else:
            self.nSamps = int(nSamps)

        try:
            if len(inc) > -1:
                self.inc = np.array(inc)
        except:
            self.inc = np.array([inc])

        try:
            if len(posAng) > -1:
                self.posAng = np.array(posAng)
        except:
            self.posAng = np.array([posAng])

        try:
            if len(gasSigma) > -1:
                self.gasSigma = np.array(gasSigma)
        except:
            self.gasSigma = np.array([gasSigma])

        try:
            if len(diskThick) > -1:
                self.diskThick = np.array(diskThick)
        except:
            self.diskThick = np.array([diskThick])

        try:
            if len(sbProf) > -1:
                self.sbProf = np.array(sbProf)
        except:
            self.sbProf = np.array([sbProf])

        try:
            if len(sbRad) > -1:
                self.sbRad = np.array(sbRad)
        except:
            self.sbRad = np.array([sbRad])

        try:
            if len(velRad) > -1:
                self.velRad = np.array(velRad)
        except:
            self.velRad = np.array([velRad])

        try:
            if len(velProf) > -1:
                self.velProf = np.array(velProf)
        except:
            self.velProf = np.array([velProf])

        try:
            if len(flux_clouds) > -1:
                self.flux_clouds = np.array(flux_clouds)
        except:
            self.flux_clouds = np.array([flux_clouds])

        try:
            if len(vRadial) > 1:
                self.vRadial = np.array(vRadial)
            else:
                self.vRadial = np.array([0])
        except:
            self.vRadial = np.array([vRadial])

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

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

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def makebeam(self, xpixels, ypixels, beamSize, cellSize=1, cent=None):

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

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def kinms_sampleFromArbDist_oneSided(self, sbRad, sbProf, nSamps, diskThick, fixSeed=None):

        #start = time.time()

        if self.verbose: print('Generating cloudlets,', end=' ')

        # If variables are not entered by user, adopt default (global) values.
        if not fixSeed:
            seed = np.random.uniform(0, 100, 4).astype('int')
        else:
            seed = self.seed
        
        # Randomly generate the radii of clouds based on the distribution given by the brightness profile.
        px = scipy.integrate.cumtrapz(sbProf * 2 * np.pi * abs(sbRad), abs(sbRad), initial=0) #  Integrates the surface brightness profile
        px /= max(px) # Normalised integral of the surface brightness profile
        rng1 = np.random.RandomState(seed[0])
        pick = rng1.random_sample(nSamps) # Draws random float samples in the range [0,1]
        interpfunc = interpolate.interp1d(px,sbRad, kind='linear') # Interplolate (x,y) -returns interpolated radii based on SBprof
        r_flat = interpfunc(pick)
        
        # Generates a random phase around the galaxy's axis for each cloud.
        rng2 = np.random.RandomState(seed[1])
        phi = rng2.random_sample(nSamps) * 2 * np.pi

        # Find the thickness of the disk at the radius of each cloud.
        if len(diskThick) > 1 and len(diskThick) != len(sbRad):
            print('\n \n ... Please make sure the length of diskThick is the same as that of sbRad! Returning.')
            return

        elif len(diskThick) > 1:
            diskThick = np.array(diskThick)
            interpfunc2 = interpolate.interp1d(sbRad, diskThick, kind='linear')
            diskThick_here = interpfunc2(r_flat)
            if self.verbose: print('using the scale height profile provided.')
        else:
            diskThick_here = diskThick
            if self.verbose: print('using a constant scale height of ' + str(diskThick) + '.')

        # Generates a random (uniform) z-position satisfying |z|<disk_here.
        rng3 = np.random.RandomState(seed[3])
        z_pos = diskThick_here * rng3.uniform(-1, 1, nSamps)

        # Calculate the x & y position of the clouds in the x-y plane of the disk.
        r_3d = np.sqrt((r_flat ** 2) + (z_pos ** 2))
        theta = np.arccos(z_pos / r_3d)
        x_pos = ((r_3d * np.cos(phi) * np.sin(theta)))
        y_pos = ((r_3d * np.sin(phi) * np.sin(theta)))

        # Generates the output array
        inClouds = np.empty((nSamps, 3))
        inClouds[:,0] = x_pos
        inClouds[:,1] = y_pos
        inClouds[:,2] = z_pos
               
        #end = time.time()
        #duration = end-start
        #print('kinms_sampleFromArbDist_oneSided duration: ', duration)

        return inClouds

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#
    
    def kinms_create_velField_oneSided(self, velRad, posAng_rad=None, inc_rad=None):

        #start = time.time()

        if not self.fixSeed:
            seed = np.random.uniform(0, 100, 4).astype('int')
        else:
            seed = self.seed
                                                                
        velInterFunc = interpolate.interp1d(velRad, self.velProf, kind='linear')  # Interpolate the velocity profile as a function of radius
        
        vRad = velInterFunc(self.r_flat)  # Evaluate the velocity profile at the sampled radii

        # Calculate a peculiar velocity for each cloudlet based on the velocity dispersion
        rng4 = np.random.RandomState(seed[3]) 
        velDisp = rng4.randn(len(self.x_pos))

        if len(self.gasSigma) > 1:
            gasSigmaInterFunc = interpolate.interp1d(velRad, self.gasSigma, kind='linear')
            velDisp *= gasSigmaInterFunc(self.r_flat)
        else:
            velDisp *= self.gasSigma
                
        # Find the rotation angle so the velocity field has the correct position angle (allows warps)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   REMOVAL OF POSITION ANGLE OFFSET BETWEEN SBPROF AND VELPROF  ~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
        
        if not self.vPosAng:
            ang2rot = 0

        elif len(self.vPosAng) > 1:
            vPosAngInterFunc = interpolate.interp1d(velRad, self.vPosAng, kind='linear') #posiiton angle as a function of radius
            vPosAng_rad = vPosAngInterFunc(self.r_flat) ### DO WE NEED TO CALCULATE ANG2ROT FOR THIS IF AS WELL AS THOSE BELOW? !!! ###

        else:
            vPosAng_rad = np.full(len(self.r_flat), self.vPosAng)
            ang2rot = ((posAng_rad - vPosAng_rad))

        #Calculate the los velocity for each cloudlet
        los_vel = velDisp                                                                                                                    
        los_vel += (-1) * vRad * (np.cos(np.arctan2((self.y_pos + self.vPhaseCent[1]),
                (self.x_pos + self.vPhaseCent[0])) + (np.radians(ang2rot))) * np.sin(np.radians(inc_rad)))

        #Add radial inflow/outflow
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   OPTIONAL INFLOW/OUTFLOW TO THE DISK    ~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

        if len(self.vRadial) > 1:
            vRadialInterFunc = interpolate.interp1d(velRad, self.vRadial, kind='linear')
            vRadial_rad = vRadialInterFunc(self.r_flat)
        else:
            vRadial_rad = np.full(len(self.r_flat), self.vRadial)

        los_vel += vRadial_rad * (np.sin(np.arctan2((self.y_pos + self.vPhaseCent[1]),
                (self.x_pos + self.vPhaseCent[0])) + (np.radians(ang2rot))) * np.sin(np.radians(inc_rad)))

        # Output the array of los velocities
        #end = time.time()
        #duration = end-start
        #print('kinms_create_velField_oneSided: ', duration)
        
        return los_vel

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def save_fits(self, cube, cent):

        hdu = fits.PrimaryHDU(cube.T)

        hdu.header['CDELT1'] = self.cellSize / -3600
        hdu.header['CDELT2'] = self.cellSize / 3600
        hdu.header['CDELT3'] = self.dv * 1000
        hdu.header['CRPIX1'] = cent[0] - 1
        hdu.header['CRPIX2'] = cent[1] - 1
        hdu.header['CRPIX3'] = cent[2]
        hdu.header['CRVAL1'] = self.ra or "None given" 
        hdu.header['CRVAL2'] = self.dec or "None given"
        hdu.header['CRVAL3'] = (self.vSys * 1000.) or "None given", 'm/s'
        hdu.header['CUNIT1'] = 'deg'
        hdu.header['CUNIT2'] = 'deg'
        hdu.header['CUNIT3'] = 'm/s'
        hdu.header['BSCALE'] = 1
        hdu.header['BZERO'] = 0
        hdu.header['BMIN'] = self.beamSize[0] / 3600
        hdu.header['BMAJ'] = self.beamSize[1] / 3600
        hdu.header['BTYPE'] = 'Intensity'
        hdu.header['BPA'] = self.beamSize[2]
        hdu.header['CTYPE1'] = 'RA---SIN'
        hdu.header['CTYPE2'] = 'DEC--SIN'
        hdu.header['CTYPE3'] = 'VRAD'
        hdu.header['EQUINOX'] = 2000
        hdu.header['RADESYS'] = 'FK5'
        hdu.header['BUNIT'] = 'Jy/beam'
        hdu.header['SPECSYS'] = 'BARYCENT'

        hdu.writeto(self.fileName + '_simcube.fits', overwrite=True, output_verify='fix')

        return

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def gasGravity_velocity(self, x_pos, y_pos, z_pos, massDist, velRad):

        if not len(massDist) == 2:
            print('\n Please provide "massDist" as a list of [gasmass, distance] - total gas mass in solar masses, total distance in Mpc. Returning.')
            return

        grav_const = 4.301e-3  # g in solar masses, pc, and km/s
        arcsec_to_pc = 4.84  # Angular distance in arcsec to physical distance in pc, when seen at distance D in Mpc
        
        x_pos = np.array(x_pos); y_pos = np.array(y_pos); z_pos = np.array(z_pos)
        massDist = np.array(massDist); velRad = np.array(velRad)
        
        rad = np.sqrt((x_pos**2) + (y_pos**2) + (z_pos**2))  # 3D radius
        cumMass = ((np.arange(x_pos.size + 1)) * (massDist[0] / x_pos.size))  # Cumulative mass

        max_velRad = np.max(velRad).clip(min=np.max(rad), max=None) + 1  # The max vel_Rad clipped to above the minimum rad
        new_rad = np.insert(sorted(rad), 0, 0)  # sorts rad and puts a 0 value at the start of it

        cumMass_interFunc = interpolate.interp1d(new_rad, cumMass, kind='linear', fill_value='extrapolate')  # Interpolates the cumulative mass as a function of radii

        add_to_circ_vel = np.sqrt(grav_const * cumMass_interFunc(velRad)) / (arcsec_to_pc * velRad * massDist[1])
        add_to_circ_vel[~np.isfinite(add_to_circ_vel)] = 0

        return add_to_circ_vel

    #=========================================================================#
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~HELP FUNCTIONS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    #=========================================================================#

    def generate_cloudlets(self):
        """
        Generate cloudlets if inClouds is not defined.
        """

        if not len(self.sbRad) or not len(self.sbProf):
            print('\nPlease define either \"inClouds\" or \"sbRad\" and \"sbProf\". Returning.')
            return
        else:
            self.inClouds_given = False
            self.inClouds = self.kinms_sampleFromArbDist_oneSided(self.sbRad, self.sbProf, self.nSamps,
                                                                  self.diskThick,
                                                                  self.fixSeed)

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def set_cloud_positions(self):
        """
        Calculate and return the positions and velocities of the cloudlets in inClouds,
        and the radial distance in the x and y plane.
        """

        self.x_pos = (self.inClouds[:, 0] / self.cellSize)
        self.y_pos = (self.inClouds[:, 1] / self.cellSize)
        self.z_pos = (self.inClouds[:, 2] / self.cellSize)
        self.r_flat = np.sqrt(self.x_pos ** 2 + self.y_pos ** 2)

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def create_warp(self, array, r_flat):
        '''
        If the array provided is an array, create a warp. If it's a single value, create a flat profile.
        '''

        if len(array) > 1:
            interp_func = interpolate.interp1d(self.velRad, array, kind='linear')
            radial_profile = interp_func(r_flat * self.cellSize)

        else:
            radial_profile = np.full(len(r_flat), array)

        return radial_profile

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def inclination_projection(self, inc_rad, x1, y1, z1):
        """
        Project the clouds to take into account inclination.
        """
        c = np.cos(np.radians(inc_rad))
        s = np.sin(np.radians(inc_rad))
        x2 = x1
        y2 = (c * y1) + (s * z1)
        z2 = (-s * y1) + (c * z1)

        return x2, y2, z2

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def position_angle_rotation(self, ang, x2, y2, z2):
        """
        Correct orientation by rotating by position angle.
        """

        c = np.cos(np.radians(ang))
        s = np.sin(np.radians(ang))
        x3 = (c * x2) + (s * y2)
        y3 = (-s * x2) + (c * y2)
        z3 = z2

        return x3, y3, z3

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def set_cloud_velocities(self):

        # Find the los velocity and cube position of the clouds
        # If los velocity specified, assume that the clouds have already been projected correctly.
        if len(self.vLOS_clouds):
            los_vel = self.vLOS_clouds
            x3 = self.x_pos
            y3 = self.y_pos
            z3 = self.z_pos

        # If los velocities not specified, calculate them.
        # Include the potential of the gas.
        elif not len(self.velProf):
            print('\nPlease define either \"vLOS_clouds\" or \"velRad\" and \"velProf\". Returning.')
            return

        else:
            # If velRad is not defined but sbRad is, set velRad to sbRad
            if not len(self.velRad) and len(self.sbRad):
                print('\n"velRad" not specified, setting it to "sbRad".')
                self.velRad = self.sbRad

            if len(self.massDist) > 1:
                gasGravVel = self.gasGravity_velocity(self.x_pos * self.cellSize, self.y_pos * self.cellSize, self.z_pos * self.cellSize, self.massDist, self.velRad)
                self.velProf = np.sqrt((self.velProf ** 2) + (gasGravVel ** 2))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # ~~~   CREATION OF POSITION ANGLE/INCLINATION  WARPS IN THE DISK ~~~~~#
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            posAng_rad = self.create_warp(self.posAng, self.r_flat)
            inc_rad = self.create_warp(self.inc, self.r_flat)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # ~~~   CREATION OF LOS VELOCITIES IF NOT PROVIDED  ~~~~~~~~~~~~~~~~~~~#
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            los_vel = self.kinms_create_velField_oneSided((self.velRad / self.cellSize), posAng_rad=posAng_rad, inc_rad=inc_rad)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
            # ~~~   PROJECT CLOUDS IN POSITION ANGLE AND INCLINATION   ~~~~~~~~~~~~#
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            x2, y2, z2 = self.inclination_projection(inc_rad, self.x_pos, self.y_pos, self.z_pos)
            x3, y3, z3 = self.position_angle_rotation(posAng_rad, x2, y2, z2)

        return x3, y3, z3, los_vel

    # =========================================================================#
    # /////////////////////////////////////////////////////////////////////////#
    # =========================================================================#

    def find_clouds_in_cube(self, los_vel, cent, x2, y2, x_size, y_size, v_size):
        """
        Returns the clouds that lie inside the cube.
        """

        # Centre the clouds in the cube on the centre of the object.
        los_vel_dv_cent2 = np.round((los_vel / self.dv) + cent[2])
        x2_cent0 = np.round(x2 + cent[0])
        y2_cent1 = np.round(y2 + cent[1])

        # Find the reduced set of clouds that lie inside the cube.
        subs = np.where(((x2_cent0 >= 0) & (x2_cent0 < x_size) & (y2_cent1 >= 0) & (y2_cent1 < y_size) & \
                         (los_vel_dv_cent2 >= 0) & (los_vel_dv_cent2 < v_size)))[0]

        clouds2do = np.empty((len(subs), 3))
        clouds2do[:, 0] = x2_cent0[subs]
        clouds2do[:, 1] = y2_cent1[subs]
        clouds2do[:, 2] = los_vel_dv_cent2[subs]

        return clouds2do, subs

    # =========================================================================#
    # /////////////////////////////////////////////////////////////////////////#
    # =========================================================================#

    def add_fluxes(self, clouds2do, subs, x_size, y_size, v_size):
        """
        If there are clouds to use, and we know the flux of each cloud, add them to the cube.
        If not, bin each position to get a relative flux.
        """

        nsubs = len(subs)

        if nsubs > 0:

            if len(self.flux_clouds) > 1:

                if not self.inClouds_given:
                    print('\n\"flux_clouds\" can only be used in combination with \"inClouds\". '
                          'Please specify \"inClouds\" if you would like to define \"flux_clouds\". Returning.')
                    return

                if not (len(self.flux_clouds.shape) == 1 and len(self.flux_clouds) == max(self.inClouds.shape)):
                    print('\nPlease make sure \"flux_clouds\" is a 1D array matching the length of \"inClouds\". '
                          'Returning.')
                    return

                cube = np.zeros((np.int(x_size), np.int(y_size), np.int(v_size)))
                self.flux_clouds = self.flux_clouds[subs]

                x = clouds2do[:, 0].astype('int')
                y = clouds2do[:, 1].astype('int')
                z = clouds2do[:, 2].astype('int')

                cube[(x, y, z)] = self.flux_clouds

            else:
                cube, edges = np.histogramdd(clouds2do, bins=(x_size, y_size, v_size),
                                             range=((0, x_size), (0, y_size), (0, v_size)))

        else:
            cube = np.zeros((np.int(x_size), np.int(y_size), np.int(v_size)))

        return cube

        # =========================================================================#
        # /////////////////////////////////////////////////////////////////////////#
        # =========================================================================#

    def normalise_cube(self, cube, psf):
        """
        Normalise cube by the known integrated flux.
        """

        if self.intFlux > 0:
            if not self.cleanOut:
                cube *= ((self.intFlux * psf.sum()) / (cube.sum() * self.dv))
            else:
                cube *= ((self.intFlux) / (cube.sum() * self.dv))

        elif len(self.flux_clouds) > 0:
            cube *= (self.flux_clouds.sum() / cube.sum())

        else:
            cube /= cube.sum()

        return cube

    def model_cube(self):
            
        """
                                                       
        # Set all values that were not defined by user to default values and make sure the right values get printed
        local_vars = locals()
        global_vars = vars(self)
        print_dict = {} # 0 = user defined, 1 = default, 2 = bool
        
        #start =  time.time()

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

        if self.verbose:
            self.print_variables(print_dict)
            
        """
                
        # Work out images sizes
        x_size = np.round(self.xs / self.cellSize)
        y_size = np.round(self.ys / self.cellSize)
        v_size = np.round(self.vs / self.dv)

        cent = [(x_size / 2) + (self.phaseCent[0] / self.cellSize), (y_size / 2) + (self.phaseCent[1] / self.cellSize),
                (v_size / 2) + (self.vOffset / self.dv)]
        
        self.vPhaseCent = self.vPhaseCent / [self.cellSize, self.cellSize]

        # If cloudlets not previously specified, generate them
        if len(self.inClouds) < 1:
            self.generate_cloudlets()

        self.set_cloud_positions()
        x2, y2, z2, los_vel = self.set_cloud_velocities()
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   POPULATE THE CUBE AND FIND NON-ZERO ELEMENTS   ~~~#~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # Find the clouds inside the cube
        clouds2do, subs = self.find_clouds_in_cube(los_vel, cent, x2, y2, x_size, y_size, v_size)

        # Add fluxes to the clouds
        cube = self.add_fluxes(clouds2do, subs, x_size, y_size, v_size)
            
        # Convolve with the beam point spread function to obtain a dirty cube
        if not self.cleanOut:

            psf = self.makebeam(x_size, y_size, self.beamSize)

            for i in range(cube.shape[2]):
                if np.sum(cube[:, :, i]) > 0:
                    cube[:, :, i] = convolve_fft(cube[:, :, i], psf)

        # Normalise the cube by known integrated flux
        self.normalise_cube(cube, psf)
                
        # If appropriate, generate the FITS file header and save to disc.
        if self.fileName:
            self.save_fits(cube, cent)

        # Plot the results if so desired
        if self.toplot:
            KinMS_plotter(cube, self.xs, self.ys, self.vs, self.cellSize, self.dv, self.beamSize).makeplots()

        # Output the final cube
        if self.returnClouds:
            retClouds = np.empty((self.nSamps, 3))
            retClouds[:, 0] = x2 * self.cellSize
            retClouds[:, 1] = y2 * self.cellSize
            retClouds[:, 2] = z2 * self.cellSize

            return cube, retClouds, los_vel

        else:
            return cube
    
    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#
    
#=============================================================================#
#/// END OF CLASS ////////////////////////////////////////////////////////////#
#=============================================================================#


#KinMS().gasGravity_velocity([3,2,1], [3,2,1], [3,2,1], [10, 10], [10,5,40])

#KinMS(20, 20, 40, 2, 10, 2, 30, sbRad=[30,40,50], sbProf=[1,2,3], velProf=[5,10,15]).model_cube()


#=============================================================================#
#/// END OF SCRIPT ///////////////////////////////////////////////////////////#
#=============================================================================#


