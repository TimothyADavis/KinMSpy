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

import time
import matplotlib.pyplot as plt

#=============================================================================#
#/// START OF CLASS //////////////////////////////////////////////////////////#
#=============================================================================#

class KinMS:
        
    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#

    def __init__(self, xs, ys, vs, cellSize, dv, beamSize, inc, posAng, gasSigma=0, diskThick=0, flux_clouds=0, 
                 sbProf=[], sbRad=[], velRad=[], velProf=[], inClouds=[], vLOS_clouds=[], massDist=[], ra=None, 
                 dec=None, nSamps=None, fixedSeed=None, intFlux=None, vSys=None, phaseCent=None, vOffset=None, 
                 vRadial=None, vPosAng=None, vPhaseCent=None, restFreq=None, fileName=False, fixSeed=False, 
                 cleanOut=False, returnClouds=False, verbose=False):
                        
        self.xs = xs
        self.ys = ys
        self.vs = vs
        self.cellSize = cellSize
        self.dv = dv
        self.beamSize = beamSize
        self.inc = np.array(inc)
        self.posAng = np.array(posAng)
        self.gasSigma = np.array(gasSigma)        
        self.diskThick = np.array(diskThick)
        self.flux_clouds = np.array(flux_clouds)
        self.sbProf = np.array(sbProf)
        self.sbRad = np.array(sbRad)                 
        self.velRad = np.array(velRad) 
        self.velProf = np.array(velProf) 
        self.inClouds = np.array(inClouds)
        self.vLOS_clouds = np.array(vLOS_clouds) 
        self.massDist = np.array(massDist)
        self.ra = ra 
        self.dec = dec 
        self.nSamps = nSamps or 5e5
        self.fixedSeed = fixedSeed or np.array([100, 101, 102, 103])
        self.intFlux = intFlux or 0
        self.vSys = vSys
        self.phaseCent = phaseCent or np.zeros(2)
        self.vOffset = vOffset or 0        
        self.vRadial = vRadial or 0
        self.vPosAng = vPosAng or 0
        self.vPhaseCent = vPhaseCent or np.zeros(2)
        self.restFreq = restFreq or 115.271e9
        self.fileName = fileName
        self.fixSeed = fixSeed
        self.cleanOut = cleanOut
        self.returnClouds = returnClouds
        self.verbose = verbose
        
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
            seed = np.random.uniform(0,100,4)
        else:
            seed = self.fixedSeed

        # Set everything to numpy arrays to accept list input
        sbRad = np.array(sbRad)
        sbProf = np.array(sbProf)
        
        nSamps = int(nSamps)
        
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
        try:
            if len(diskThick) != len(sbRad):
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
        inClouds = np.empty((nSamps, 3))
        inClouds[:,0] = xPos
        inClouds[:,1] = yPos
        inClouds[:,2] = zPos
               
        #end = time.time()
        #duration = end-start
        #print('kinms_sampleFromArbDist_oneSided duration: ', duration)

        return inClouds

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#
    
    def kinms_create_velField_oneSided(self, velRad, velProf, r_flat, inc, posAng, gasSigma, xPos, yPos, fixSeed=None,
                                       vPhaseCent=None, vRadial=None, posAng_rad=None, inc_rad=None, vPosAng=None):
            
        #start = time.time()
                        
        if not fixSeed:
            seed = np.random.uniform(0,100,4)
        else:
            seed = self.fixedSeed
                                                                
        velInterFunc = interpolate.interp1d(velRad, velProf, kind='linear') # Interpolate the velocity profile as a function of radius
        
        vRad = velInterFunc(r_flat) # Evaluate the velocity profile at the sampled radii
                
        # Calculate a peculiar velocity for each cloudlet based on the velocity dispersion
        rng4 = np.random.RandomState(seed[3]) 
        velDisp = rng4.randn(len(xPos))
        try:
                if len(gasSigma) > 1:
                    gasSigmaInterFunc = interpolate.interp1d(velRad,gasSigma,kind='linear')
                    velDisp *= gasSigmaInterFunc(r_flat)
                else:
                    velDisp *= gasSigma
        except:
                velDisp *= gasSigma
                
        # Find the rotation angle so the velocity field has the correct position angle (allows warps)
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   REMOVAL OF POSITION ANGLE OFFSET BETWEEN SBPROF AND VELPROF  ~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
        
        if not vPosAng:
            ang2rot=0
        else:
            try:
                if len(vPosAng) > 1:
                    vPosAngInterFunc = interpolate.interp1d(velRad,vPosAng,kind='linear') #posiiton angle as a function of radius
                    vPosAng_rad = vPosAngInterFunc(r_flat) ### DO WE NEED TO CALCULATE ANG2ROT FOR THIS IF AS WELL AS THOSE BELOW? !!! ###
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
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   OPTIONAL INFLOW/OUTFLOW TO THE DISK    ~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
        
        try:
            if len(vRadial) > 1:
                vRadialInterFunc = interpolate.interp1d(velRad,vRadial,kind='linear')
                vRadial_rad = vRadialInterFunc(r_flat)
            else:
                vRadial_rad = np.full(len(r_flat), vRadial, np.double)
        except:
            vRadial_rad=np.full(len(r_flat),vRadial,np.double)
        los_vel += vRadial_rad * (np.sin(np.arctan2((yPos+vPhaseCent[1]),(xPos + vPhaseCent[0])) + (np.radians(ang2rot))) * np.sin(np.radians(inc_rad)))
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

    def gasGravity_velocity(self, xPos, yPos, zPos, massDist, velRad):

        if not len(massDist) == 2:
            print('\n Please provide "massDist" as a list of [gasmass, distance] - total gas mass in solar masses, total distance in Mpc. Returning.')
            return

        grav_const = 4.301e-3  # g in solar masses, pc, and km/s
        arcsec_to_pc = 4.84  # Angular distance in arcsec to physical distance in pc, when seen at distance D in Mpc
        
        xPos = np.array(xPos); yPos = np.array(yPos); zPos = np.array(zPos)
        massDist = np.array(massDist); velRad = np.array(velRad)
        
        rad = np.sqrt((xPos**2) + (yPos**2) + (zPos**2))  # 3D radius
        cumMass = ((np.arange(xPos.size + 1)) * (massDist[0] / xPos.size))  # Cumulative mass

        max_velRad = np.max(velRad).clip(min=np.max(rad), max=None) + 1  # The max vel_Rad clipped to above the minimum rad
        new_rad = np.insert(sorted(rad), 0, 0)  # sorts rad and puts a 0 value at the start of it

        cumMass_interFunc = interpolate.interp1d(new_rad, cumMass, kind='linear', fill_value='extrapolate')  # Interpolates the cumulative mass as a function of radii

        add_to_circ_vel = np.sqrt(grav_const * cumMass_interFunc(velRad)) / (arcsec_to_pc * velRad * massDist[1])
        add_to_circ_vel[~np.isfinite(add_to_circ_vel)] = 0

        return add_to_circ_vel

    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#
          
    def model_cube(self):
                        
        if len(self.fixedSeed):
                self.fixSeed = True
            
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
                
        """
                
        # Check to see if the input is a list and convert to array if so        
        xs = np.array(self.xs)
        ys = np.array(self.ys)
        vs = np.array(self.vs)
                
        """

        self.__dict__.update(global_vars)

        if self.verbose:
            self.print_variables(print_dict)
            
        """

        # Set variables to numpy arrays if necessary
        self.sbRad = np.array(self.sbRad)
        self.sbProf = np.array(self.sbProf)
        self.velProf = np.array(self.velProf)
        self.velRad = np.array(self.velRad)
                
        # Work out images sizes
        xSize = np.round(xs / self.cellSize)
        ySize = np.round(ys / self.cellSize)
        vSize = np.round(vs / self.dv)

        cent = [(xSize / 2) + (self.phaseCent[0] / self.cellSize), (ySize / 2) + (self.phaseCent[1] / self.cellSize),
                (vSize / 2) + (self.vOffset / self.dv)]
        
        vPhaseCent = self.vPhaseCent / [self.cellSize, self.cellSize]

        # If cloudlets not previously specified, generate them
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~   CREATING CLOUDLETS  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        
        inClouds_given = True
        if not len(self.inClouds):
            if not len(self.sbRad) or not len(self.sbProf):
                print('\nPlease define either \"inClouds\" or \"sbRad\" and \"sbProf\". Returning.')
                return
            else:
                inClouds_given = False
                self.inClouds = self.kinms_sampleFromArbDist_oneSided(self.sbRad, self.sbProf, self.nSamps,
                                                                      self.diskThick, self.fixSeed) ### FIXSEED IS FALSE HERE, DO WE WANT THAT AS A DEFAULT? ###

        xPos = (self.inClouds[:, 0] / self.cellSize)
        yPos = (self.inClouds[:, 1] / self.cellSize)
        zPos = (self.inClouds[:, 2] / self.cellSize)
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
        elif not len(self.velProf):
            print('\nPlease define either \"vLOS_clouds\" or \"velRad\" and \"velProf\". Returning.')
            return
    
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   CHANGE VELOCITY PROFILE BASED ON POTENTIAL OF INNER GAS ~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

        else:

            # If velRad is not defined but sbRad is, set velRad to sbRad
            if not len(self.velRad) and len(self.sbRad):
                print('\n"velRad" not specified, setting it to "sbRad".')
                self.velRad = self.sbRad

            if len(self.massDist)>1:
                gasGravVel = self.gasGravity_velocity(xPos * self.cellSize, yPos * self.cellSize, zPos * self.cellSize, self.massDist, self.velRad)
                self.velProf = np.sqrt((self.velProf ** 2) + (gasGravVel ** 2))
                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   CREATION OF POSITION ANGLE WARPS IN THE DISK ~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
        
            if isinstance(self.posAng,(int,np.ndarray)): # Needed for differentiating int and array input
                    self.posAng = 0-self.posAng 

            try:
                if len(self.posAng) > 1:
                    # Creating a warp
                    posAngRadInterFunc = interpolate.interp1d(self.velRad, self.posAng, kind='linear') # Interpolation of position angles wrt cloudlet radii
                    posAng_rad = posAngRadInterFunc(r_flat * self.cellSize)
                else:
                    # No warp 
                    posAng_rad = np.full(len(r_flat), self.posAng, float)
            except:
                # No warp
                posAng_rad = np.full(len(r_flat), self.posAng, float)
                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#      
        #~~~   CREATION OF INCLINATION  WARPS IN THE DISK ~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

            try:
                if len(self.inc) > 1:
                    # Creating a warp in inclination
                    incRadInterFunc = interpolate.interp1d(self.velRad, self.inc, kind='linear')
                    inc_rad = incRadInterFunc(r_flat * self.cellSize)
                else:
                    # No inclination warp
                    inc_rad = np.full(len(r_flat), self.inc, float)
            except:
                # No inclination warp
                inc_rad = np.full(len(r_flat), self.inc, float)
                            
            # Calculate the LOS velocity.
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   CREATION OF LOS VELOCITIES IF NOT PROVIDED  ~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        

            los_vel = self.kinms_create_velField_oneSided((self.velRad / self.cellSize), self.velProf, r_flat, self.inc, \
                      self.posAng, self.gasSigma, xPos, yPos, fixSeed=self.fixSeed, vPhaseCent= vPhaseCent, \
                      vRadial = self.vRadial, posAng_rad=posAng_rad, inc_rad=inc_rad, vPosAng=self.vPosAng) 
            
           
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#      
        #~~~   PROJECT CLOUDS IN POSITION ANGLE AND INCLINATION   ~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#            
            
            # Project the clouds to take into account inclination.
            c = np.cos(np.radians(inc_rad))
            s = np.sin(np.radians(inc_rad))
            x2 = xPos
            y2 = (c * yPos) + (s * zPos)
            z2 = (-s * yPos) + (c * zPos)

            # Correct orientation by rotating by position angle.
            ang = posAng_rad
            c = np.cos(np.radians(ang))
            s = np.sin(np.radians(ang))
            x3 = (c * x2) + (s * y2)
            y3 = (-s * x2) + (c * y2)
            x2 = x3
            y2 = y3
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   POPULATE THE CUBE AND FIND NON-ZERO ELEMENTS   ~~~#~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

        # Now add the flux into the cube.
        # Centre the clouds in the cube on the centre of the object.
        los_vel_dv_cent2 = np.round((los_vel / self.dv) + cent[2])
        x2_cent0 = np.round(x2 + cent[0])
        y2_cent1 = np.round(y2 + cent[1])

        # Find the reduced set of clouds that lie inside the cube.
        subs = np.where(((x2_cent0 >= 0) & (x2_cent0 < xSize) & (y2_cent1 >= 0) & (y2_cent1 < ySize) & \
                         (los_vel_dv_cent2 >= 0) & (los_vel_dv_cent2 < vSize)))[0]

        nsubs = len(subs)
        
        clouds2do = np.empty((nsubs, 3))
        clouds2do[:,0] = x2_cent0[subs]
        clouds2do[:,1] = y2_cent1[subs]
        clouds2do[:,2] = los_vel_dv_cent2[subs]
        
        # If there are clouds to use, and we know the flux of each cloud, add them to the cube.
        # If not, bin each position to get a relative flux.
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   ADD IN FLUX VALUES TO THE CUBE OR BIN FOR RELATIVE FLUX   ~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

        if nsubs > 0:
                
            try:
                if len(self.flux_clouds) > 1:
                    self.flux_clouds = np.array(self.flux_clouds)

                    if not inClouds_given:
                        print('\n\"flux_clouds\" can only be used in combination with \"inClouds\". '
                              'Please specify \"inClouds\" if you would like to define \"flux_clouds\". Returning.')
                        return

                    if not (len(self.flux_clouds.shape) == 1 and len(self.flux_clouds) == max(self.inClouds.shape)):
                        print('\nPlease make sure \"flux_clouds\" is a 1D array matching the length of \"inClouds\". '
                              'Returning.')
                        return

                    cube = np.zeros((np.int(xSize), np.int(ySize), np.int(vSize)))
                    self.flux_clouds = self.flux_clouds[subs]

                    x = clouds2do[:, 0].astype('int')
                    y = clouds2do[:, 1].astype('int')
                    z = clouds2do[:, 2].astype('int')

                    cube[(x,y,z)] = self.flux_clouds

                else:
                    print('\nPlease use \"intFlux\" to define the total flux in the cube. Returning.')
                    return

            except:
                cube, edges = np.histogramdd(clouds2do, bins=(xSize, ySize, vSize),
                                             range=((0,xSize), (0, ySize), (0, vSize)))

                ### THIS IS WHERE THE CUBE IS COMING OUT BLANK!!! ###
                ### ADD IN PRINT STATEMENT (DON'T USE THIS WITHOUT WHAT?) ###

        else:
            cube = np.zeros((np.int(xSize), np.int(ySize), np.int(vSize)))
            
            
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   CONVOLVE CUBE WITH THE BEAM   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
            
        # Convolve with the beam point spread function to obtain a dirty cube
        if not self.cleanOut:

            psf = self.makebeam(xSize, ySize, self.beamSize)

            for i in range(cube.shape[2]):
                if np.sum(cube[:, :, i]) > 0:
                    cube[:, :, i] = convolve_fft(cube[:, :, i], psf)

        # Normalise by the known integrated flux.
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   NORMALISE BY SOME KNOWN INTEGRATED FLUX   ~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
        
        if self.intFlux > 0:
            if not self.cleanOut:
                cube *= ((self.intFlux * psf.sum()) / (cube.sum() * self.dv))
            else:
                cube *= ((self.intFlux) / (cube.sum() * self.dv))

        else:
            try:
                len(self.flux_clouds) > 0
                cube *= (self.flux_clouds.sum() / cube.sum())
            except:
                cube /= cube.sum()
                
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#       
        #~~~   OUTPUT CUBE TO FITS FILE   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 
                
        # If appropriate, generate the FITS file header and save to disc.
        if self.fileName:
            self.save_fits(cube, cent)

        # Output the final cube
        if self.returnClouds:
            retClouds = np.empty((self.nSamps, 3))
            retClouds[:, 0] = x2 * self.cellSize
            retClouds[:, 1] = y2 * self.cellSize
            retClouds[:, 2] = z2 * self.cellSize
            
            #end = time.time()
            #duration = end-start
            #print('model_cube duration: ', duration)     

            return cube, retClouds, los_vel

        else:
            #end = time.time()
            #duration = end-start
            #print('model_cube duration: ', duration) 
            return cube
    
    #=========================================================================#
    #/////////////////////////////////////////////////////////////////////////#
    #=========================================================================#
    
#=============================================================================#
#/// END OF CLASS ////////////////////////////////////////////////////////////#
#=============================================================================#
    


"""
RUN EVERY TEST IN THE TEST SUITE AND COMPARE TIME TO TIM'S CODE
"""

#KinMS().gasGravity_velocity([3,2,1], [3,2,1], [3,2,1], [10, 10], [10,5,40])

#KinMS(20, 20, 40, 2, 10, 2, 30, sbRad=[30,40,50], sbProf=[1,2,3], velProf=[5,10,15]).model_cube()


#=============================================================================#
#/// END OF SCRIPT ///////////////////////////////////////////////////////////#
#=============================================================================#


