# coding: utf-8
# Radial motion profiles/parameterisations from Spekkens & Sellwood (2007)
import numpy as np

class radial_motion:
    def __init__(self):
        pass

    
    class bisymmetric_flow:
        def __init__(self,radius,v2t,v2r,phib):
            self.v2t=v2t
            self.v2r=v2r
            self.phib=phib
            self.radius=radius
            
            
        def __call__(self,r_flatv,theta,inc_rad):
            radial_vel=np.interp(r_flatv, self.radius, self.v2r)
            transverse_vel=np.interp(r_flatv, self.radius, self.v2t)       
            theta_b=theta-np.deg2rad(self.phib)
            return (-1)*(transverse_vel*np.cos(2*theta_b)*np.cos(theta) + (radial_vel*np.sin(2*theta_b)*np.sin(theta))) * np.sin(np.radians(inc_rad))
            
    class lopsided_flow:
        def __init__(self,radius,v1t,v1r,phib):
            self.v1t=v1t
            self.v1r=v1r
            self.phib=phib
            self.radius=radius
            
            
        def __call__(self,r_flatv,theta,inc_rad):
            radial_vel=np.interp(r_flatv, self.radius, self.v1r)
            transverse_vel=np.interp(r_flatv, self.radius, self.v1t)
            theta_b=theta-np.deg2rad(self.phib)
            return (-1)*(transverse_vel*np.cos(theta_b)*np.cos(theta) + (radial_vel*np.sin(theta_b)*np.sin(theta))) * np.sin(np.radians(inc_rad))
                    
            
            
    class pure_radial:
        def __init__(self,radius,vradial):
            self.vradial=vradial
            self.radius=radius
            
            
        def __call__(self,r_flatv,theta,inc_rad):
            radial_vel=np.interp(r_flatv, self.radius, self.vradial)                               
            return (radial_vel*np.sin(theta)) * np.sin(np.radians(inc_rad))