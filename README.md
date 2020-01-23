<img style="float:top,right" src="utils/logo_files/Logo.png" width="400">

[![Documentation Status](https://readthedocs.org/projects/kinmspy/badge/?version=latest)](https://kinmspy.readthedocs.io/en/latest/?badge=latest)

The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions. The routines are written with flexibility in mind, and have been used in various different applications, including investigating the kinematics of molecular gas in early-type galaxies (Davis et al, MNRAS, Volume 429, Issue 1, p.534-555, 2013), and determining supermassive black-hole masses from CO interfermetric observations (Davis et al., Nature, 2013). They are also useful for creating input datacubes for further simulation in e.g. CASA's sim_observe tool.

If you find any bugs, or wish to be kept up to date when new versions of this software are released, please email me us at DavisT -at- cardiff.ac.uk, Zabelnj -at- cardiff.ac.uk, Dawsonj5 -at- cardiff.ac.uk

To run the test suite, which demonstrates some of the functionality of this code, please checkout/download the code, and navigate to the directory. Then:
```
ipython
from KinMS_testsuite import *
run_tests()
```

KinMS_testsuite.py contains the example code, which can be modified and updated for most use cases. If you want to use KinMSpy for fitting, see [KinMSpy_MCMC](https://github.com/TimothyADavis/kinmspy_mcmc) for an example of interfacing with Bayesian fitting codes.


Many thanks,

Dr Timothy A. Davis, Nikki Zabel, and James M. Dawson

Cardiff, UK
