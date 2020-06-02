from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

 
setup(name='kinms',
       version='2.0.2',
       description='The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions.',
       url='https://github.com/TimothyADavis/KinMSpy',
       author='Timothy A. Davis',
       author_email='DavisT@cardiff.ac.uk',
       long_description=long_description,
       long_description_content_type="text/markdown",
       license='MIT',
       packages=['kinms','kinms.examples','kinms.utils'],
       install_requires=[
           'numpy',
           'matplotlib',
           'scipy',
           'astropy',
       ],
       classifiers=[
         'Development Status :: 4 - Beta',
         'License :: OSI Approved :: MIT License',
         'Programming Language :: Python :: 3',
         'Operating System :: OS Independent',
       ],
       zip_safe=False)
       
