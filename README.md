<img style="float:top,right" src="kinms/docs/Logo.png" width="400">


[![Documentation Status](https://readthedocs.org/projects/kinmspydocs/badge/?version=latest)](https://kinmspydocs.readthedocs.io/en/latest/?badge=latest) [![Python 3.6](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-382/) [![PyPI version](https://badge.fury.io/py/kinms.svg)](https://badge.fury.io/py/kinms) 

The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions. The routines are written with flexibility in mind, and have been used in various different applications, including investigating the kinematics of molecular gas in early-type galaxies ([Davis et al, MNRAS, Volume 429, Issue 1, p.534-555, 2013](https://academic.oup.com/mnras/article/429/1/534/1022845)), and determining supermassive black-hole masses from CO interfermetric observations (Davis et al., Nature, 2013). They are also useful for creating input datacubes for further simulation in e.g. [CASA](https://casa.nrao.edu/)'s sim_observe tool.


### Install

KinMSpy is designed with Python users in mind. Lots of work has gone into making it lightweight and fast. You can use it in the same way that you would use [NumPy](https://numpy.org/) or [Astropy](https://www.astropy.org/) etc. You can install KinMS with `pip install kinms`. Alternatively you can download the code, navigate to the directory you unpack it too, and run `python setup.py install`.
    
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

To get you started fitting observations with KinMS, see the walk through here: [Example fitting tutorial](https://github.com/TimothyADavis/KinMSpy/blob/master/kinms/docs/KinMSpy_tutorial.ipynb)


### Upgrading from version 1 

Unlike previous generations of KinMS, version 2.0+ uses Python classes for a more modular and adjustable experience. Plotting routines can be changed and cube modelling can be probed at different stages if required. The main change you will need if upgrading from version 1.0 is to change all calls to `KinMS(...)` to `KinMS(...).model_cube()`. The tutorial notebooks above have full details of the new features.

### Commumication

If you find any bugs, or wish to be kept up to date when new versions of this software are released, please raise an issue here on github, or email us at DavisT -at- cardiff.ac.uk, Zabelnj -at- cardiff.ac.uk, Dawsonj5 -at- cardiff.ac.uk

### License

KinMSpy is MIT-style licensed, as found in the LICENSE file.


Many thanks,

Dr Timothy A. Davis, Nikki Zabel, and James M. Dawson

Cardiff, UK
