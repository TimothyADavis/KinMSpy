import numpy as np
import scipy.integrate
from scipy import interpolate, fftpack
from astropy.io import fits
from astropy.convolution import convolve_fft
import makebeam

def kinms_samplefromarbdist_onesided(sbrad,sbprof,nsamps,fixseed,diskthick=0):
    px=np.zeros(len(sbprof))
    sbprof=sbprof*(2*np.pi*abs(sbrad))  
    px=np.cumsum(sbprof)
    px/=max(px)           
    np.random.seed(fixseed[0])               
    pick=np.random.random(nsamps)  
    np.random.seed(fixseed[1])        
    phi=np.random.random(nsamps)*2*np.pi     
    interpfunc = interpolate.interp1d(px,sbrad, kind='linear')
    r_flat=interpfunc(pick)
    if isinstance(diskthick, (list, tuple, np.ndarray)):
        interpfunc2 = interpolate.interp1d(sbrad,diskthick,kind='linear')
        diskthick_here=interpfunc2(r_flat)
    else:
        diskthick_here=diskthick    
    np.random.seed(fixseed[2])      
    zpos=diskthick_here*np.random.uniform(-1,1,nsamps) 
    r_3d = np.sqrt((r_flat**2)+(zpos**2))                                                               
    theta=np.arccos(zpos/r_3d)                                                              
    xpos=((r_3d*np.cos(phi)*np.sin(theta)))                                                        
    ypos=((r_3d*np.sin(phi)*np.sin(theta)))
    inclouds=np.empty((nsamps,3))
    inclouds[:,0]=xpos
    inclouds[:,1]=ypos
    inclouds[:,2]=zpos                                                          
    return inclouds                                                               

def kinms_create_velfield_onesided(velrad,velprof,r_flat,inc,posang,gassigma,seed,xpos,ypos,vphasecent=[0,0],vposang=False,vradial=0,posang_rad=0,inc_rad=0):
    velinterfunc = interpolate.interp1d(velrad,velprof,kind='linear')
    vrad=velinterfunc(r_flat)
    los_vel=np.empty(len(vrad))
    np.random.seed(seed[3])
    veldisp=np.random.randn(len(xpos)) 
    if isinstance(gassigma, (list, tuple, np.ndarray)):
        gassigmainterfunc = interpolate.interp1d(velrad,gassigma,kind='linear')
        veldisp*=gassigmainterfunc(r_flat)
    else:
        veldisp*=gassigma
    if isinstance(vradial, (list, tuple, np.ndarray)):
        vradialinterfunc = interpolate.interp1d(velrad,vradial,kind='linear')
        vradial_rad=vradialinterfunc(r_flat)
    else:
        vradial_rad=np.full(len(r_flat),vradial)
    if not vposang:
        ang2rot=0.0
    else:
        if isinstance(vposang, (list, tuple, np.ndarray)):
            vposanginterfunc = interpolate.interp1d(velrad,vposang,kind='linear')
            vposang_rad=vposanginterfunc(r_flat)
        else:
            vposang_rad=np.full(len(r_flat),vposang)
        ang2rot=((posang_rad-vposang_rad))
    los_vel=veldisp                                                                                                                    
    los_vel+=(-1)*vrad*(np.cos(np.arctan2((ypos+vphasecent[1]),(xpos+vphasecent[0]))+(np.radians(ang2rot)))*np.sin(np.radians(inc_rad)))           
    if vradial != 0:
        if isinstance(vradial, (list, tuple, np.ndarray)):
            vradialinterfunc = interpolate.interp1d(velrad,vradial,kind='linear')
            vradial_rad=vradialinterfunc(r_flat)
        else:
            vradial_rad=np.full(len(r_flat),vradial)
        los_vel+=vradial_rad*(np.sin(np.arctan2((ypos+vphasecent[1]),(xpos+vphasecent[0]))+(np.radians(ang2rot)))*np.sin(np.radians(inc_rad)))
    return los_vel






def KinMS(xs,ys,vs,dx,dy,dv,beamsize,inc,gassigma=0,sbprof=[],sbrad=[],velrad=[],velprof=[],galname=False,diskthick=0,cleanout=False,ra=0,dec=0,nsamps=1e5,posang=0.0,intflux=0,inclouds=[],vlos_clouds=[],flux_clouds=0,vsys=0,restfreq=115.271e9,phasecen=np.array([0.,0.]),voffset=0,fixseed=False,vradial=0,vposang=0,vphasecen=np.array([0.,0.])):
    
    if fixseed:
        fixseed=[100,101,102,103]
    else:
        fixseed=np.random.randint(0,100,4)
    
    if not isinstance(beamsize, (list, tuple, np.ndarray)):
        beamsize=np.array([beamsize,beamsize,0])
    
    # work out images sizes
    xsize = float(round(xs/dx))
    ysize = float(round(ys/dy))
    vsize = float(round(vs/dv))
    cent=[(xsize/2.)+(phasecen[0]/dx),(ysize/2.)+(phasecen[1]/dy),(vsize/2.)+(voffset/dv)]
    vphasecent=(vphasecen-phasecen)/[dx,dy]
    
    
    if not len(inclouds):
        inclouds=kinms_samplefromarbdist_onesided(sbrad,sbprof,nsamps,fixseed,diskthick=diskthick)
    xpos=(inclouds[:,0]/dx)
    ypos=(inclouds[:,1]/dy)
    zpos=(inclouds[:,2]/dx)
    r_flat=np.sqrt((xpos*xpos) + (ypos*ypos))
        
    if len(vlos_clouds):
        los_vel=vlos_clouds
        x2=xpos
        y2=ypos
        z2=zpos
    else:     
        posang=90-posang
        if isinstance(posang, (list, tuple, np.ndarray)):
            posangradinterfunc = interpolate.interp1d(velrad,posang,kind='linear')
            posang_rad=posangradinterfunc(r_flat)
        else:
            posang_rad=np.full(len(r_flat),posang)
        
        if isinstance(inc, (list, tuple, np.ndarray)):
            incradinterfunc = interpolate.interp1d(velrad,inc,kind='linear')
            inc_rad=incradinterfunc(r_flat)
        else:
            inc_rad=np.full(len(r_flat),inc)
        
        los_vel=kinms_create_velfield_onesided(velrad,velprof,r_flat,inc,posang,gassigma,fixseed,xpos,ypos,vphasecent=vphasecent,vposang=vposang,vradial=vradial,inc_rad=inc_rad,posang_rad=posang_rad)
        c = np.cos(np.radians(inc_rad))
        s = np.sin(np.radians(inc_rad))
        x2 =  xpos
        y2 =  c*ypos + s*zpos
        z2 = -s*ypos + c*zpos       
        
        ang=posang_rad
        c = np.cos(np.radians(ang))
        s = np.sin(np.radians(ang))
        x3 =  c*x2 + s*y2
        y3 = -s*x2 + c*y2
        x2=x3
        y2=y3
    # now add the flux into the cube
    los_vel_dv_cent2=np.round((los_vel/dv)+cent[2])
    x2_cent0=np.round(x2+cent[0])
    y2_cent1=np.round(y2+cent[1])
    subs = np.where(((x2_cent0 >= 0) & (x2_cent0 < xsize) & (y2_cent1 >= 0) & (y2_cent1 < ysize) & (los_vel_dv_cent2 >= 0) & (los_vel_dv_cent2 < vsize)))
    nsubs=subs[0].size
    clouds2do=np.empty((nsubs,3))
    clouds2do[:,0]=x2_cent0[subs]
    clouds2do[:,1]=y2_cent1[subs]
    clouds2do[:,2]=los_vel_dv_cent2[subs]
    
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
    
    
    if not cleanout:
       psf=makebeam.beam(xsize,ysize,[beamsize[0]/dx,beamsize[1]/dy],rot=beamsize[2])
       w2do=np.where(cube.sum(axis=0).sum(axis=0) >0)[0]
       for i in range(0,w2do.size): cube[:,:,w2do[i]]=convolve_fft(cube[:,:,w2do[i]], psf)
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
            
    if galname:
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
        hdu.writeto(galname+"_simcube.fits",clobber=True,output_verify='fix')
    return cube