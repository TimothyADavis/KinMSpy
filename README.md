The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions. The routines are written with flexibility in mind, and have been used in various different applications, including investigating the kinematics of molecular gas in early-type galaxies (e.g. Davis et al, MNRAS, Volume 429, Issue 1, p.534-555, 2013), and determining supermassive black-hole masses from CO interfermetric observations (e.g. Davis et al., Nature, 2013). They are also useful for creating input datacubes for further simulation in e.g. CASA's sim_observe tool.

If you find any bugs, or wish to be kept up to date when new versions of this software are released, please email me at DavisT -at- cardiff.ac.uk

### Install
You can install KinMS with `pip install kinms`
    
It requires the following modules:

* numpy
* matplotlib
* scipy
* astropy

### Documentation


To run the test suite, which demonstrates some of the functionality of this code, please install the software. Then:
```
ipython
from kinms.examples.KinMS_testsuite import *
run_tests()
```
examples/KinMS_testsuite.py contains the example code, which can be modified and updated for most use cases.

To get you started fitting galaxies with KinMS, see the walk through here: https://github.com/TimothyADavis/KinMSpy/blob/master/kinms/documentation/KinMSpy_tutorial.ipynb


Author & License
-----------------

Copyright 2019 Timothy A. Davis

Built by Timothy A. Davis <https://github.com/TimothyADavis>. Licensed under the MIT License (see ``LICENSE``).