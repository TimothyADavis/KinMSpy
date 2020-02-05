import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kinmspytest", 
    version="2.0.8",
    author="KinMS_Team",
    author_email="dawsonj5@cardiff.ac.uk",
    description="Kinematic molecular simulation tool for astronomers",
    long_description="The KinMS (KINematic Molecular Simulation) package can be used to simulate observations of arbitary molecular/atomic cold gas distributions. The routines are written with flexibility in mind, and have been used in various different applications, including investigating the kinematics of molecular gas in early-type galaxies (Davis et al, MNRAS, Volume 429, Issue 1, p.534-555, 2013), and determining supermassive black-hole masses from CO interfermetric observations (Davis et al., Nature, 2013). They are also useful for creating input datacubes for further simulation in e.g. CASA's sim_observe tool.",
    long_description_content_type="text/markdown",
    url="https://github.com/NikkiZabel/KinMSpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
