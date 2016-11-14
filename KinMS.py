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
from makebeam import makebeam


def kinms_samplefromarbdist_onesided(sbrad,sbprof,nsamps,seed,diskthick=0.0):
    """

    This function takes the input radial distribution and generates the positions of
    `nsamps` cloudlets from under it. It also accounts for disk thickness
    if requested. Returns 
    
    Parameters
    ----------
    sbrad : np.ndarray of double
            Radius vector (in units of pixels).
    
    sbprof : np.ndarray of double
            Surface brightness profile (arbitrarily scaled).
    
    nsamps : int
            Number of samples to draw from the distribution.
    
    seed : list of int
            List of length 4 containing the seeds for random number generation.
    
    diskthick : double or np.ndarray of double
         (Default value = 0.0)
            The disc scaleheight. If a single value then this is used at all radii.
            If a ndarray then it should have the same length as sbrad, and will be 
            the disc thickness as a function of sbrad. 

    Returns
    -------
    inclouds : np.ndarray of double
            Returns an ndarray of `nsamps` by 3 in size. Each row corresponds to
            the x, y, z position of a cloudlet. 
    """
    #Randomly generate the radii of clouds based on the distribution given by the brightness profile
    px=np.zeros(len(sbprof))
    sbprof=sbprof*(2*np.pi*abs(sbrad))  
    px=np.cumsum(sbprof)
    px/=max(px)           
    np.random.seed(seed[0])               
    pick=np.random.random(nsamps)  
    interpfunc = interpolate.interp1d(px,sbrad, kind='linear')
    r_flat=interpfunc(pick)
    
    #Generates a random phase around the galaxy's axis for each cloud
    np.random.seed(seed[1])        
    phi=np.random.random(nsamps)*2*np.pi     
 
    # Find the thickness of the disk at the radius of each cloud
    if isinstance(diskthick, (list, tuple, np.ndarray)):
        interpfunc2 = interpolate.interp1d(sbrad,diskthick,kind='linear')
        diskthick_here=interpfunc2(r_flat)
    else:
        diskthick_here=diskthick    
    
    #Generates a random (uniform) z-position satisfying |z|<disk_here 
    np.random.seed(seed[2])      
    zpos=diskthick_here*np.random.uniform(-1,1,nsamps) 
    
    #Calculate the x & y position of the clouds in the x-y plane of the disk
    r_3d = np.sqrt((r_flat**2)+(zpos**2))                                                               
    theta=np.arccos(zpos/r_3d)                                                              
    xpos=((r_3d*np.cos(phi)*np.sin(theta)))                                                        
    ypos=((r_3d*np.sin(phi)*np.sin(theta)))
    
    #Generates the output array
    inclouds=np.empty((nsamps,3))
    inclouds[:,0]=xpos
    inclouds[:,1]=ypos
    inclouds[:,2]=zpos                                                          
    return inclouds                                                               

def kinms_create_velfield_onesided(velrad,velprof,r_flat,inc,posang,gassigma,seed,xpos,ypos,vphasecent=[0.0,0.0],vposang=False,vradial=0.0,posang_rad=0.0,inc_rad=0.0):
    """

    This function takes the input circular velocity distribution
    and the position of point sources and creates the velocity field 
    taking into account warps, inflow/outflow etc as required.
    
    Parameters
    ----------
    velrad : np.ndarray of double
            Radius vector (in units of pixels).
    
    velprof : np.ndarray of double
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
        
    posang : double or np.ndarray of double
            Position angle of the disc, using the usual astronomical convention.
            Can be either a double, or an array of doubles. If single valued
            then the disc major axis is straight. If an array is passed then it should
            describe how the position angle changes as a function of `velrad`.
            Used to create position angle warps.
        
    gassigma : double or np.ndarray of double
            Velocity dispersion of the gas. Units of km/s. 
            Can be either a double, or an array of doubles. If single valued
            then the velocity dispersion is constant throughout the disc.
            If an array is passed then it should describe how the velocity
            dispersion changes as a function of `velrad`.
        
    seed : list of int
            List of length 4 containing the seeds for random number generation.
    
    xpos : np.ndarray of double
            X position of each cloudlet. Units of pixels. 
    
    ypos : np.ndarray of double
            Y position of each cloudlet. Units of pixels. 
    
    vphasecent : list of double
         (Default value = [0, 0])
            Kinematic centre of the rotation in the x-y plane. Units of pixels.
            Used if the kinematic and morphological centres are not the same.
    
    vposang : double or np.ndarray of double
         (Default value = False)
            Kinematic position angle of the disc, using the usual astronomical convention.
            Can be either a double, or an array of doubles. If single valued
            then the disc kinematic major axis is straight. If an array is passed then it should
            describe how the kinematic position angle changes as a function of `velrad`.
            Used if the kinematic and morphological position angles are not the same.
    
    vradial : double or np.ndarray of double
         (Default value = 0)
            Magnitude of inflow/outflowing motions (km/s). Negative
            numbers here are inflow, positive numbers denote
            outflow. These are included in the velocity field using
            formalism of KINEMETRY (Krajnović et al. 2006 MNRAS, 366, 787). 
            Can input a constant or a vector, giving the radial
            motion as a function of the radius vector
            `velrad`. Default is no inflow/outflow.
    
    posang_rad : double or np.ndarray of double
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
    velinterfunc = interpolate.interp1d(velrad,velprof,kind='linear')
    vrad=velinterfunc(r_flat)
    los_vel=np.empty(len(vrad))
    # Calculate a peculiar velocity for each cloudlet based on the velocity dispersion
    np.random.seed(seed[3])
    veldisp=np.random.randn(len(xpos)) 
    if isinstance(gassigma, (list, tuple, np.ndarray)):
        gassigmainterfunc = interpolate.interp1d(velrad,gassigma,kind='linear')
        veldisp*=gassigmainterfunc(r_flat)
    else:
        veldisp*=gassigma
    
    #Add radial inflow/outflow (runs twice - this version is not necessary?)
    if isinstance(vradial, (list, tuple, np.ndarray)):
        vradialinterfunc = interpolate.interp1d(velrad,vradial,kind='linear')
        vradial_rad=vradialinterfunc(r_flat)
    else:
        vradial_rad=np.full(len(r_flat),vradial,np.double)
    # Find the rotation angle so the velocity field has the correct position angle (allows warps)
    if not vposang:
        ang2rot=0.0
    else:
        if isinstance(vposang, (list, tuple, np.ndarray)):
            vposanginterfunc = interpolate.interp1d(velrad,vposang,kind='linear')
            vposang_rad=vposanginterfunc(r_flat)
        else:
            vposang_rad=np.full(len(r_flat),vposang,np.double)
        ang2rot=((posang_rad-vposang_rad))
    #Calculate the los velocity for each cloudlet
    los_vel=veldisp                                                                                                                    
    los_vel+=(-1)*vrad*(np.cos(np.arctan2((ypos+vphasecent[1]),(xpos+vphasecent[0]))+(np.radians(ang2rot)))*np.sin(np.radians(inc_rad)))           
    
    #Add radial inflow/outflow
    if vradial != 0:
        if isinstance(vradial, (list, tuple, np.ndarray)):
            vradialinterfunc = interpolate.interp1d(velrad,vradial,kind='linear')
            vradial_rad=vradialinterfunc(r_flat)
        else:
            vradial_rad=np.full(len(r_flat),vradial,np.double)
        los_vel+=vradial_rad*(np.sin(np.arctan2((ypos+vphasecent[1]),(xpos+vphasecent[0]))+(np.radians(ang2rot)))*np.sin(np.radians(inc_rad)))
    # Output the array of los velocities
    return los_vel






def KinMS(xs,ys,vs,dx,dy,dv,beamsize,inc,gassigma=0,sbprof=[],sbrad=[],velrad=[],velprof=[],filename=False,diskthick=0,cleanout=False,ra=0,dec=0,nsamps=100000,posang=0.0,intflux=0,inclouds=[],vlos_clouds=[],flux_clouds=0,vsys=0,restfreq=115.271e9,phasecen=np.array([0.,0.]),voffset=0,fixseed=False,vradial=0,vposang=0,vphasecen=np.array([0.,0.])):
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
    
    dx : float
        Pixel size in x-direction (arcsec/pixel)
    
    dy : float
        Pixel size in y-direction (arcsec/pixel)
    
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
    
    """
    
    
    nsamps=int(nsamps)
    # Generate seeds for use in future calculations
    if fixseed:
        fixseed=[100,101,102,103]
    else:
        fixseed=np.random.randint(0,100,4)
    
    # If beam profile not fully specified, generate it:
    if not isinstance(beamsize, (list, tuple, np.ndarray)):
        beamsize=np.array([beamsize,beamsize,0])
    
    # work out images sizes
    xsize = float(round(xs/dx))
    ysize = float(round(ys/dy))
    vsize = float(round(vs/dv))
    cent=[(xsize/2.)+(phasecen[0]/dx),(ysize/2.)+(phasecen[1]/dy),(vsize/2.)+(voffset/dv)]
    vphasecent=(vphasecen-phasecen)/[dx,dy]

    #If cloudlets not previously specified, generate them
    if not len(inclouds):
        inclouds=kinms_samplefromarbdist_onesided(sbrad,sbprof,nsamps,fixseed,diskthick=diskthick)
    xpos=(inclouds[:,0]/dx)
    ypos=(inclouds[:,1]/dy)
    zpos=(inclouds[:,2]/dx)
    r_flat=np.sqrt((xpos*xpos) + (ypos*ypos))
    
    #Find the los velocity and cube position of the clouds
    if len(vlos_clouds):
        los_vel=vlos_clouds
        x2=xpos
        y2=ypos
        z2=zpos
    else:     
        # As los velocities not specified, calculate them
        # Setup to project onto the line of sight
        posang=90-posang
        if isinstance(posang, (list, tuple, np.ndarray)):
            posangradinterfunc = interpolate.interp1d(velrad,posang,kind='linear')
            posang_rad=posangradinterfunc(r_flat)
        else:
            posang_rad=np.full(len(r_flat),posang,np.double)
        
        if isinstance(inc, (list, tuple, np.ndarray)):
            incradinterfunc = interpolate.interp1d(velrad,inc,kind='linear')
            inc_rad=incradinterfunc(r_flat)
        else:
            inc_rad=np.full(len(r_flat),inc,np.double)
        
        # Calculate the los velocity and cube position of the clouds
        los_vel=kinms_create_velfield_onesided(velrad/dx,velprof,r_flat,inc,posang,gassigma,fixseed,xpos,ypos,vphasecent=vphasecent,vposang=vposang,vradial=vradial,inc_rad=inc_rad,posang_rad=posang_rad)
        c = np.cos(np.radians(inc_rad))
        s = np.sin(np.radians(inc_rad))
        x2 =  xpos
        y2 =  c*ypos + s*zpos
        z2 = -s*ypos + c*zpos       
        
        # Correct orientation by rotating by position angle
        ang=posang_rad
        c = np.cos(np.radians(ang))
        s = np.sin(np.radians(ang))
        x3 =  c*x2 + s*y2
        y3 = -s*x2 + c*y2
        x2=x3
        y2=y3
    # now add the flux into the cube
    # Centre the clouds in the cube on the centre of the object
    los_vel_dv_cent2=np.round((los_vel/dv)+cent[2])
    x2_cent0=np.round(x2+cent[0])
    y2_cent1=np.round(y2+cent[1])
    
    #Find the reduced set of clouds that lie inside the cube
    subs = np.where(((x2_cent0 >= 0) & (x2_cent0 < xsize) & (y2_cent1 >= 0) & (y2_cent1 < ysize) & (los_vel_dv_cent2 >= 0) & (los_vel_dv_cent2 < vsize)))
    nsubs=subs[0].size
    clouds2do=np.empty((nsubs,3))
    clouds2do[:,0]=x2_cent0[subs]
    clouds2do[:,1]=y2_cent1[subs]
    clouds2do[:,2]=los_vel_dv_cent2[subs]
    
    # If there are clouds to use, and we know the flux of each cloud, add them to the cube. If not, bin each position to get
    # a relative flux
    if nsubs > 0:
        if not isinstance(flux_clouds, (list, tuple, np.ndarray)):
            cube,edges=np.histogramdd(clouds2do,bins=(xsize,ysize,vsize),range=((0,xsize),(0,ysize),(0,vsize)))
        else:
            cube = np.zeros((xsize,ysize,vsize))
            flux_clouds=flux_clouds[subs]
            for i in range(0, nsubs):
                const = flux_clouds[i]
                csub = (clouds2do[i,0],clouds2do[i,1],clouds2do[i,2])
                cube[csub] = cube[csub] + const
    
    # Convolve with the beam point spread function to obtain a dirty cube
    if not cleanout:
       psf=makebeam(xsize,ysize,[beamsize[0]/dx,beamsize[1]/dy],rot=beamsize[2])
       w2do=np.where(cube.sum(axis=0).sum(axis=0) >0)[0]
       for i in range(0,w2do.size): cube[:,:,w2do[i]]=convolve_fft(cube[:,:,w2do[i]], psf)
    # Normalise by the known integrated flux
    if intflux > 0:
        if not cleanout:
            cube *= ((intflux*psf.sum())/(cube.sum()*dv))
        else: 
            cube *= ((intflux)/(cube.sum()*dv))
    else:
        if isinstance(flux_clouds, (list, tuple, np.ndarray)):
            cube*=(flux_clouds.sum()/cube.sum()) 
        else:
            cube/=cube.sum()
    
    # If appropriate, generate the FITS file header and save to disc
    if filename:
        hdu = fits.PrimaryHDU(cube.T)
        hdu.header['CDELT1']=(dx)/(-3600.0)
        hdu.header['CDELT2']=(dy)/3600.0
        hdu.header['CDELT3']=(dv)*1000.0
        hdu.header['CRPIX1']=(cent[0]-1)
        hdu.header['CRPIX2']=(cent[1])
        hdu.header['CRPIX3']=(cent[2])
        hdu.header['CRVAL1']=(ra)
        hdu.header['CRVAL2']=(dec)
        hdu.header['CRVAL3']=(vsys*1000.0),"m/s"
        hdu.header['CUNIT1']='deg'
        hdu.header['CUNIT2']='deg'
        hdu.header['CUNIT3']='m/s'
        hdu.header['BSCALE']=1.0             
        hdu.header['BZERO']=0.0                                                                                     
        hdu.header['BMIN']=np.min(np.array(beamsize[0:1])/3600.0)
        hdu.header['BMAJ']=np.max(np.array(beamsize[0:1])/3600.0)
        hdu.header['BTYPE']='Intensity'  
        hdu.header['BPA']=beamsize[2]
        hdu.header['CTYPE1']='RA---SIN' 
        hdu.header['CTYPE2']='DEC--SIN'
        hdu.header['CTYPE3']='VRAD'  
        hdu.header['EQUINOX']=2000.0
        hdu.header['RADESYS']='FK5'
        hdu.header['BUNIT']='Jy/beam'
        hdu.header['SPECSYS']='BARYCENT'
        hdu.writeto(filename+"_simcube.fits",clobber=True,output_verify='fix')
        
    # Output the final cube
    return cube
