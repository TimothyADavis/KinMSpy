<img style="float:top,right" src="kinms/docs/Logo.png" width="400">


[![Documentation Status](https://readthedocs.org/projects/kinmspydocs/badge/?version=latest)](https://kinmspydocs.readthedocs.io/en/latest/?badge=latest) [![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/) [![PyPI version](https://badge.fury.io/py/kinms.svg)](https://badge.fury.io/py/kinms) 
[![ASCL](https://img.shields.io/badge/ascl-2006.003-blue.svg?colorB=262255)](http://ascl.net/2006.003)


The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions. The routines are written with flexibility in mind, and have been used in various different applications, including investigating the kinematics of molecular gas in early-type galaxies ([Davis et al, MNRAS, Volume 429, Issue 1, p.534-555, 2013](https://academic.oup.com/mnras/article/429/1/534/1022845)), and determining supermassive black-hole masses from CO interfermetric observations (e.g. Davis et al., Nature, 2013). They are also useful for creating input datacubes for further simulation in e.g. [CASA](https://casa.nrao.edu/)'s sim_observe tool.


### Install

KinMSpy is designed with Python users in mind. Lots of work has gone into making it lightweight and fast. You can install KinMS with `pip install kinms`. Alternatively you can download the code, navigate to the directory you unpack it too, and run `python setup.py install`.
    
It requires the following modules:

* numpy
* matplotlib
* scipy
* astropy


### Documentation

A simple iPython notebook tutorial on the basics of KinMS can be found here: [KinMS simple tutorial](https://github.com/TimothyADavis/KinMSpy/blob/master/kinms/docs/KinMS_example_notebook.ipynb)

A further suite of examples can be found in examples/KinMS_testsuite.py, which can be modified and updated for most use cases. To run these tests you can run the following commands from within python:

```
from kinms.examples.KinMS_testsuite import *
run_tests()
```

To get you started fitting observations with KinMS, check out KinMS_fitter, which wraps KinMS and automates many tasks for you! Check it out here: [KinMS_fitter](https://github.com/TimothyADavis/KinMS_fitter).


If you need go through the nuts and bolts of fitting, see the walk through here: [Example fitting tutorial](https://github.com/TimothyADavis/KinMSpy/blob/master/kinms/docs/KinMSpy_tutorial.ipynb)

### New: KinMS 3.0 is here!
As of August 2022 KinMS 3.0 has been released. This version has subtantial speed improvements, being around 35% faster in my tests. However, to implement this the interface has had to slightly change. Now only the interferometer/observation parameters are passed to the KinMS class at instantiation, while the parameters that can change (e.g. surface brightness/velocity profiles) are passed in the `model_cube()` call. The tutorial notebooks/examples have been updated to reflect this. If you *need* to keep using the old version then it can still be imported as `from kinms import KinMS2`, although this dual support will be removed in future updates.


### New non-circular motions capability

As of version 2.2.0 KinMS now has the capability to model lopsided and bisymmetric gas flows, in addition to the pure radial motions included previously. To get started with this you need to add `from kinms.radial_motion import radial_motion`, and then pass one of the new methods to KinMS with the `radial_motion_func` keyword. `radial_motion.lopsided_flow` and `radial_motion.bisymmetric_flow` both take four arguments (a radial vector, the transverse and radial velocity as a function of that radius, and an angle for the perterbation). `radial_motion.pure_radial` replicates previous funcationality, and requires two arguments (a radius vector, and a vector for the radial velocity as a function of radius). E.g. if previously you were passing `inflowVel=inflowVel` then this would now equate to `radial_motion_func=radial_motion.pure_radial(radius,inflowVel)`.

### Communication

If you find any bugs, or wish to be kept up to date when new versions of this software are released, please raise an issue here on github, or email us at DavisT -at- cardiff.ac.uk, Zabelnj -at- cardiff.ac.uk, Dawsonj5 -at- cardiff.ac.uk

### License

KinMSpy is MIT-style licensed, as found in the LICENSE file.


Many thanks,

Dr Timothy A. Davis, Nikki Zabel, and James M. Dawson

Cardiff, UK
