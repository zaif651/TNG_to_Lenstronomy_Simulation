# TNG_to_Lenstronomy_Simulation
Set of notebooks and files for extracting data of galaxies from the IllustrisTNG cosmological suite and use for lensing simulations


Notebook Explanations:


## Illustris-information-extraction.ipynb
Notebook for extracting various information from an Illustris Galaxy

## kappa-to-lenstronomy.ipynb
Notebook for taking saved convergence/kappa file computed from TNG galaxy (z=0.5) and use in simulating lensed image for given redhsift (z=1.5 i think?) and source galaxy parameters.
Not successful :(

## TNG-lensing-potential.ipynb
Notebook for extracting galaxy information from IllustrisTNG using the API and then computing the 3D and integrated 2D potentials of the galaxy using all the stellar, dark matter, and gas particles


## 2d_potential-to-lenstronomy-sim.ipynb
Notebook that takes saved 2D and 3D potentials of a galaxy at redshift z = 0.3 along with Cartesian grid information, computes the lensing potential and then simulates the lensed image for a source at a given redshift (z=1.2) and parameters.
No success in using the INTERPOL class successfully yet :(

## debug_interpol.py
## interpol_troubleshooting_guide.py
## run_interpol_debug.py

Dedugging the INTERPOL class and Lensing potentials. I'm still testing these scripts!!

## kappa_oberved.npy, x_coord.npy, y_coord.npy
files containing kappa (lensing convergence) data

## Lenstronomy_kappa_custom.ipynb
Notebook for comparing custom lensing potential calculation with Lenstonomy's in-built function for pre-defined models for diagnosing the INTERPOL class. My function fails :(. There are simplified assumptions in the lensing potential calculation through Fourier Transform that are liking causing the issues.

Ignore these files : 

## Illustris-to-kappa.py
Script for computing convergence from stellar mass density projection only. Ignore this file for now.

## Illustris-kappa-lenstronomy-combined.py
One complete integrated pipeline from importing the Illustris Galaxy to kappa calculation to simulating lensed images. Some flaws in the script (only takes stellar density that is). Ignore this file.

## total-density-to-Lenstronomy.ipynb
Notebook for converting 3D mass density to convergence for Lenstronomy I think. It's incomplete however. Ignore this file.

