# TNG_to_Lenstronomy_Simulation
Set of notebooks and files for extracting data of galaxies from the IllustrisTNG cosmological suite and use for lensing simulations


Notebook Explanations:

## 2d_potential-to-lenstronomy-sim.ipynb
Notebook that takes saved 2D and 3D potentials of a galaxy at redshift z = 0.3 along with Cartesian grid information, computes the lensing potential and then simulates the lensed image for a source at a given redshift (z=1.2) and parameters.

## TNG-lensing-potential.ipynb
Notebook for extracting galaxy information from IllustrisTNG using the API and then computing the 3D and integrated 2D potentials of the galaxy using all the stellar, dark matter, and gas particles

## debug_interpol.py
## interpol_troubleshooting_guide.py
## run_interpol_debug.py

Dedugging the INTERPOL class and Lensing potentials. I'm still testing these scripts

## Illustris-information-extraction.ipynb
Notebook for extracting various information from an Illustris Galaxy

## kappa-to-lenstronomy.ipynb
Notebook for taking saved convergence/kappa file computed from TNG galaxy (z=0.5) and use in simulating lensed image for given redhsift (z=1.5 i think?) and source galaxy parameters.

