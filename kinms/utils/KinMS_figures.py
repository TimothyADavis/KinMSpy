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

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy import ndimage
from astropy.nddata.utils import Cutout2D
from kinms.utils.sauron_colormap import sauron
import warnings; warnings.filterwarnings("ignore", module="matplotlib")

#=============================================================================#
#/// START OF CLASS //////////////////////////////////////////////////////////#
#=============================================================================#

class KinMS_plotter:

    def __init__(self, f, xsize, ysize, vsize, cellsize, dv, beamSize, posang=None, pvdthick=None,
                 savepath=None, savename=None, pdf=True, overcube=False, title=False,rms=0.0,rmsfac=1.0):
        
        """
        :class KinMS_plotter:
            Generates moment maps and position velocity diagrams for input spectral cubes
        
        :param f:
            (numpy array) Spectral cube used to generate plots
        :param xsize:
            (float or int) x-axis size for resultant cube (in arcseconds)
        :param ysize:
            (float or int) y-axis size for resultant cube (in arcseconds)
        :param vsize:
            (float or int) Velocity axis size for resultant cube (in km/s)
        :param cellSize:
            (float or int) Pixel size required (arcsec/pixel)
        :param dv:
            (float or int) Channel size in velocity direction (km/s/channel)
        :param beamSize:
            (float or int, or list or array of float or int) Scalar or three element list for size of convolving beam (in arcseconds). If a scalar then beam is
            assumed to be circular. If a list/array of length two. these are the sizes of the major and minor axes,
            and the position angle is assumed to be 0. If a list/array of length 3, the first 2 elements are the
            major and minor beam sizes, and the last the position angle (i.e. [bmaj, bmin, bpa]).
        :param posAng:
            (float or int) Position angle (PA) of the disc (a PA of zero means that the redshifted part of the cube is aligned
            with the positive y-axis). If single valued then the disc major axis is straight. If an array is passed
            then it should describe how the position angle changes as a function of `velrad` (so this can be used
            to create position angle warps).
        :param pvdthick:
            UNDER CONSTRUCTION
        :param savepath:
            (string) path to directory in which the plots are saved to
        :param savename:
            (string) a filename assigned to the plots when saved
        :param pdf:
            (bool) Optional, default value is True.
            saves the plots as a .pdf file
        :param overcube:
            (bool) UNDER CONSTRUCTION
        :param title:
            (bool) Optional, default value is False.
            assigns a title to the plots
        """

        self.f = f
        self.xsize = xsize
        self.ysize = ysize
        self.vsize = vsize
        self.cellsize = cellsize
        self.dv = dv
        self.posang = posang or 0  
        self.pvdthick = pvdthick or 2
        self.savepath = savepath or None
        self.savename = savename or None
        self.pdf = pdf
        self.overcube = overcube
        self.title = title
        self.mask=None
        self.rms=rms
        self.rmsfac=rmsfac
        
        beamSize = np.array(beamSize)
        try:
            if len(beamSize) == 2:
                beamSize = np.append(beamSize, 0)
            if beamSize[1] > beamSize[0]:
                beamSize[1], beamSize[0] = beamSize[0], beamSize[1]
            if beamSize[2] >= 180:
                beamSize[2] -= 180
        except:
            beamSize = np.array([beamSize, beamSize, 0])
        
        self.beamsize = beamSize
            
        
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['font.family'] = 'Latin Modern Roman'
        matplotlib.rcParams.update({'font.size': 20})
        matplotlib.rcParams['legend.fontsize'] = 17.5
        matplotlib.rcParams['axes.linewidth'] = 1.5
        matplotlib.rcParams['xtick.labelsize'] = 20;
        matplotlib.rcParams['ytick.labelsize'] = 20
        matplotlib.rcParams['xtick.major.size'] = 10;
        matplotlib.rcParams['ytick.major.size'] = 10
        matplotlib.rcParams['xtick.major.width'] = 2;
        matplotlib.rcParams['ytick.major.width'] = 2
        matplotlib.rcParams['xtick.minor.size'] = 5;
        matplotlib.rcParams['ytick.minor.size'] = 5
        matplotlib.rcParams['xtick.minor.width'] = 1;
        matplotlib.rcParams['ytick.minor.width'] = 1
        matplotlib.rcParams['xtick.direction'] = 'in';
        matplotlib.rcParams['ytick.direction'] = 'in'
        matplotlib.rcParams['xtick.bottom'] = True
        matplotlib.rcParams['ytick.left'] = True
        params = {'mathtext.default': 'regular'}
        matplotlib.rcParams.update(params)
        matplotlib.rcParams['axes.labelsize'] = 30

    def gaussian(self, x, x0, sigma):
        return np.exp(-np.power((x - x0) / (sigma), 2) / 2)

    def makebeam(self, xpixels, ypixels, beamSize, cellSize=1, cent=None):

        if not cent: cent = [xpixels / 2, ypixels / 2]

        beamSize = np.array(beamSize)

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

        ### Trim around high values in the psf, to speed up the convolution ###

        psf[psf < 1e-5] = 0  # set all kernel values that are very low to zero

        # sum the psf in the beam major axis
        if 45 < beamSize[2] < 135:
            flat = np.sum(psf, axis=1)
        else:
            flat = np.sum(psf, axis=0)

        idx = np.where(flat > 0)[0]  # find the location of the non-zero values of the psf

        newsize = (idx[-1] - idx[0])  # the size of the actual (non-zero) beam is this

        if newsize % 2 == 0:
            newsize += 1  # add 1 pixel just in case
        else:
            newsize += 2  # if necessary to keep the kernel size odd, add 2 pixels

        trimmed_psf = Cutout2D(psf, (cent[1], cent[0]), newsize).data  # cut around the psf in the right location

        return trimmed_psf


    def smoothmask(self,cube):
        mask=ndimage.uniform_filter(cube, size=[1.5*(self.beamsize[0]/self.cellsize),1.5*(self.beamsize[0]/self.cellsize),4], mode='constant', cval=0.0)
        mask[mask < self.rms*self.rmsfac]=0
        mask[mask > 0] =1
        return mask

    def makeplots(self, **kwargs):

        # Create plot data from the cube
        self.mask=self.smoothmask(self.f)
        self.f*=self.mask
        
        mom0rot = np.sum(self.f, axis=2)

        if np.any(self.overcube):
            mom0over = self.overcube.sum(axis=2)

        x1 = np.arange(-self.xsize / 2, self.xsize / 2, self.cellsize)
        y1 = np.arange(-self.ysize / 2, self.ysize / 2, self.cellsize)
        v1 = np.arange(-self.vsize / 2, self.vsize / 2, self.dv)

        mom1 = (self.f*v1).sum(axis=2)/(self.f.sum(axis=2))

        pvdcube = self.f

        pvdcube = ndimage.interpolation.rotate(self.f, 90-self.posang, axes=(1, 0), reshape=False)

        if np.any(self.overcube):
            pvdcubeover = ndimage.interpolation.rotate(self.overcube, 90-self.posang, axes=(1, 0), reshape=False)

        pvd = pvdcube[:, np.int((self.ysize / (self.cellsize * 2)) - self.pvdthick):np.int((self.ysize / (self.cellsize * 2)) + self.pvdthick), :].sum(axis=1)
        if np.any(self.overcube):
            pvdover = pvdcubeover[:,
                      np.int((self.ysize / (self.cellsize * 2)) - self.pvdthick):
                      np.int((self.ysize / (self.cellsize * 2)) + self.pvdthick), :].sum(axis=1)

        beamtot = self.makebeam(self.xsize, self.ysize, self.beamsize, cellSize=self.cellsize).sum()

        spec = self.f.sum(axis=0).sum(axis=0) / beamtot

        if np.any(self.overcube):
            specover = self.overcube.sum(axis=0).sum(axis=0) / beamtot

        # Plot the results
        levs = v1[np.min(np.where(spec != 0)): np.max(np.where(spec != 0))]

        fig = plt.figure(figsize=(10, 10))

        # Plot the moment 0
        ax1 = fig.add_subplot(221, aspect='equal')
        ax1.contourf(x1, y1, mom0rot.T, levels=np.linspace(1, 0, num=10, endpoint=False)[::-1] * np.max(mom0rot),cmap="YlOrBr",origin="upper")
        if np.any(self.overcube):
            ax1.contour(x1, y1, mom0over.T, colors=('black'), levels=np.arange(0.1, 1.1, 0.1) * np.max(mom0over))

        if 'yrange' in kwargs: ax1.set_ylim(kwargs['yrange'])
        if 'xrange' in kwargs: ax1.set_xlim(kwargs['xrange'])

        plt.xlabel(r'Offset ($^{\prime\prime}$)');
        plt.ylabel(r'Offset ($^{\prime\prime}$)')

        # Plot moment 1
        ax2 = fig.add_subplot(222, aspect='equal')
        mom1[mom0rot < 0.1*np.max(mom0rot)]=-1e4
        ax2.contourf(x1, y1, mom1.T, levels=levs, cmap=sauron)
        plt.xlabel(r'Offset ($^{\prime\prime}$)');
        plt.ylabel(r'Offset ($^{\prime\prime}$)')
        if 'yrange' in kwargs: ax2.set_ylim(kwargs['yrange'])
        if 'xrange' in kwargs: ax2.set_xlim(kwargs['xrange'])

        # Plot PVD
        ax3 = fig.add_subplot(223)

        ax3.contourf(x1, v1, pvd.T, levels=np.linspace(1, 0, num=10, endpoint=False)[::-1] * np.max(pvd),
                     cmap="YlOrBr", aspect='auto')
        if np.any(self.overcube):
            ax3.contour(x1, v1, pvdover.T, colors='black', levels=np.arange(0.1, 1.1, 0.1) * np.max(pvdover))

        if 'vrange' in kwargs: ax3.set_ylim(kwargs['vrange'])
        if 'xrange' in kwargs: ax3.set_xlim(kwargs['xrange'])

        plt.xlabel(r'Offset ($^{\prime\prime}$)');
        plt.ylabel(r'Velocity (km s$^{-1}$)')

        # Plot spectrum
        ax4 = fig.add_subplot(224)
        if np.any(self.overcube):
            ax4.plot(v1, spec, drawstyle='steps', c='r')
            ax4.plot(v1, specover, 'k', drawstyle='steps')
        else:
            ax4.plot(v1, spec, drawstyle='steps', c='k')
            
        if 'vrange' in kwargs: ax4.set_xlim(kwargs['vrange'])
        if self.title: plt.suptitle(self.title)

        plt.ylabel('Flux');
        plt.xlabel(r'Velocity (km s$^{-1}$)')

        plt.tight_layout()
        plt.show()
        if self.savepath:
            if self.savename:
                if self.pdf:
                    plt.savefig(self.savepath + '/' + self.savename + '.pdf', bbox_inches='tight')
                else:
                    plt.savefig(self.savepath + '/' + self.savename + '.png', bbox_inches='tight')
            else:
                if self.pdf:
                    plt.savefig(self.savepath + '/' + 'KinMS_plots.pdf', bbox_inches='tight')
                else:
                    plt.savefig(self.savepath + '/' + 'KinMS_plots.png', bbox_inches='tight')
                    
#=============================================================================#
#/// END OF CLASS ////////////////////////////////////////////////////////////#
#=============================================================================#

#=============================================================================#
#/// END OF SCRIPT ///////////////////////////////////////////////////////////#
#=============================================================================#
