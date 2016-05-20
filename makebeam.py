import numpy as np

def beam(xpixels,ypixels,st_dev,rot=0):
    st_dev=st_dev[::-1]
    st_dev=np.array(st_dev)/2.355
    if np.tan(np.radians(rot)) == 0:
        dirfac=1
    else:
        dirfac=np.sign(np.tan(np.radians(rot)))   
    x,y=np.indices((xpixels,ypixels))
    x -=xpixels/2.
    y -=ypixels/2.           
    a=(np.cos(np.radians(rot))**2)/(2.0*(st_dev[0]**2)) + (np.sin(np.radians(rot))**2)/(2.0*(st_dev[1]**2))
    b=((dirfac)*(np.sin(2.0*np.radians(rot))**2)/(4.0*(st_dev[0]**2))) + ((-1*dirfac)*(np.sin(2.0*np.radians(rot))**2)/(4.0*(st_dev[1]**2)))
    c=(np.sin(np.radians(rot))**2)/(2.0*(st_dev[0]**2)) + (np.cos(np.radians(rot))**2)/(2.0*(st_dev[1]**2))
    psf=np.exp(-1*(a*(x**2) - 2.0*b*(x*y) + c*(y**2)))         
    return psf