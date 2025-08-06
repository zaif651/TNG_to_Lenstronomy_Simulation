#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 23:53:24 2025

@author: abdullahalzaif
"""

# %%
# import of standard python libraries
import numpy as np
import os
import time
import astropy.io.fits as pyfits

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


import lenstronomy


# lenstronomy imports
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.ImSim.image_model import ImageModel
import lenstronomy.Util.param_util as param_util
import lenstronomy.Util.simulation_util as sim_util
import lenstronomy.Util.image_util as image_util
from lenstronomy.Util import kernel_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

#import emcee for MCMC
import emcee

#import corner module with relevant installations for implementation
import corner

#%%

from lenstronomy.LensModel.Profiles.interpol import Interpol

kappa_obs = np.load("/Users/abdullahalzaif/Desktop/PIII_Project_Strong_Lensing/IllustrisTNG/kappa_observed.npy", allow_pickle=True)

x_coord = np.load("/Users/abdullahalzaif/Desktop/PIII_Project_Strong_Lensing/IllustrisTNG/x_coord.npy", allow_pickle=True)

y_coord = np.load("/Users/abdullahalzaif/Desktop/PIII_Project_Strong_Lensing/IllustrisTNG/y_coord.npy", allow_pickle=True)

#%%

plt.imshow(np.log10(kappa_obs.T), origin='lower', cmap='inferno', extent=[-300, 300, -300, 300])
plt.colorbar(label='Convergence κ')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title('Projected Convergence Map from Stars')
plt.show()

#%%

x_pos =[i+0.6 for i in x_coord]
y_pos =[i+0.6 for i in y_coord]

x_pos = x_pos[:500]
y_pos = y_pos[:500]


delta_x  = 1.0 #kpc
D_ang_lens = 1297.04084139 #Mpc
delta_rad = 2 * np.arctan((delta_x*10**3)/(2*D_ang_lens * 10**6)) 


Delta_pix = delta_rad * 206264.806 #arcseconds

x_ang_max, x_ang_min = 300*Delta_pix, -300*Delta_pix


#%%
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq

def kappa_to_potential(kappa_map, pixel_size_arcsec):
    """
    Convert a convergence (κ) map to lensing potential (ψ) using Fourier transforms.
    
    Parameters:
    - kappa_map: 2D numpy array of convergence values.
    - pixel_size_arcsec: pixel scale in arcseconds.

    Returns:
    - potential_map: 2D numpy array of lensing potential ψ.
    """
    n = kappa_map.shape[0]
    L = n * pixel_size_arcsec  # total angular size
    dk = 1.0 / L

    kx = fftfreq(n, d=pixel_size_arcsec)
    ky = fftfreq(n, d=pixel_size_arcsec)
    kx, ky = np.meshgrid(kx, ky)

    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0  # avoid divide by zero at DC

    kappa_ft = fft2(kappa_map)
    psi_ft = -2 * kappa_ft / (4 * np.pi**2 * k2)  # Note 4π² from FFT convention
    psi_ft[0, 0] = 0.0  # subtract mean, zero potential at DC mode

    potential_map = np.real(ifft2(psi_ft))
    return potential_map


#%%

lensing_pot = kappa_to_potential(kappa_obs, Delta_pix)

numPix = lensing_pot.shape[0]
x_angular = np.linspace(-numPix/2 * Delta_pix, numPix/2 * Delta_pix, numPix)
y_angular = np.linspace(-numPix/2 * Delta_pix, numPix/2 * Delta_pix, numPix)

#%%
plt.imshow((lensing_pot), origin='lower', cmap='inferno', extent=[-300, 300, -300, 300])
plt.colorbar(label='Convergence κ')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title('Projected Convergence Map from Stars')
plt.show()

#%%

def potential_to_deflection(potential, delta_pix):
    """
    Compute deflection angles (α_x, α_y) from lensing potential.
    """
    alpha_y, alpha_x = np.gradient(potential, delta_pix)  # dψ/dx, dψ/dy
    return alpha_x, alpha_y

#%%

alpha_x, alpha_y = potential_to_deflection(lensing_pot, Delta_pix)


# %% define lens configuration and cosmology (not for lens modelling)

z_lens = 0.5
z_source = 2.0
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.)


#%% data specifics
sigma_bkg = .05  #  background noise per pixel (Gaussian)
exp_time = 5000.  #  exposure time (arbitrary units, flux per pixel is in units #photons/exp_time unit)
numPix = 500  #  cutout pixel size
deltaPix = Delta_pix  #  pixel size in arcsec (area per pixel = deltaPix**2)
fwhm = 0.1  # full width half max of PSF (only valid when psf_type='gaussian')
psf_type = 'GAUSSIAN'  # 'GAUSSIAN', 'PIXEL', 'NONE'
kernel_size = 91

# initial input simulation

# generate the coordinate grid and image properties
kwargs_data = sim_util.data_configure_simple(numPix, deltaPix, exp_time, sigma_bkg)
data_class = ImageData(**kwargs_data)

# generate the psf variables

kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix, 'fwhm': fwhm}
psf_class = PSF(**kwargs_psf)



# %% Set up the Input Model

# lensing quantities

lens_model_list = ['INTERPOL']
kwargs_lens = [{ 'grid_interp_x': x_angular,
    'grid_interp_y': y_angular,
    'f_' : lensing_pot,
    'f_x': alpha_x,
    'f_y': alpha_y
}]


lens_model_class = LensModel(lens_model_list=lens_model_list, z_lens=z_lens, z_source=z_source, cosmo=cosmo)




# choice of source type
source_type = 'SERSIC'  # 'SERSIC' or 'SHAPELETS'

source_x =0.05
source_y = 0.1



# list of light profiles (for lens and source)
# 'SERSIC': Sersic profile of source

phi_G, q = 0.5, 0.8
#e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
kwargs_sersic_source = {'amp': 4000, 'R_sersic': 0.2, 'n_sersic': 1, 'e1': 0, 'e2': 0, 'center_x': source_x, 'center_y': source_y}
source_model_list = ['SERSIC_ELLIPSE']
kwargs_source = [kwargs_sersic_source]
source_model_class = LightModel(light_model_list=source_model_list)


# lens light model
#phi_G, q = 0.9, 0.9
#e1, e2 = param_util.phi_q2_ellipticity(phi_G, q)
kwargs_sersic_lens = {'amp': 8000, 'R_sersic': 0.4, 'n_sersic': 2., 'e1': 0, 'e2': 0, 'center_x': 0.0, 'center_y': 0}
lens_light_model_list = ['SERSIC_ELLIPSE']
kwargs_lens_light = [kwargs_sersic_lens]
lens_light_model_class = LightModel(light_model_list=lens_light_model_list)

# Lensed Image positions

lensEquationSolver = LensEquationSolver(lens_model_class)
x_image, y_image = lensEquationSolver.findBrightImage(source_x, source_y, kwargs_lens, numImages=4,
                                                      min_distance=deltaPix, search_window=numPix * deltaPix)
mag = lens_model_class.magnification(x_image, y_image, kwargs=kwargs_lens)
kwargs_ps = [{'ra_image': x_image, 'dec_image': y_image,
                           'point_amp': np.abs(mag)*1000}]  # quasar point source position in the source plane and intrinsic brightness
point_source_list = ['LENSED_POSITION']
point_source_class = PointSource(point_source_type_list=point_source_list, fixed_magnification_list=[False])

kwargs_numerics = {'supersampling_factor': 1}

imageModel = ImageModel(data_class, psf_class, lens_model_class, source_model_class,
                                lens_light_model_class, point_source_class, kwargs_numerics=kwargs_numerics)

# generate image
image_sim = imageModel.image(kwargs_lens, kwargs_source, kwargs_lens_light, kwargs_ps)
poisson = image_util.add_poisson(image_sim, exp_time=exp_time)
bkg = image_util.add_background(image_sim, sigma_bkd=sigma_bkg)
image_sim = image_sim + bkg + poisson

#%%

data_class.update_data(image_sim)
kwargs_data['image_data'] = image_sim


kwargs_model = {'lens_model_list': lens_model_list,
                 'lens_light_model_list': lens_light_model_list,
                 'source_light_model_list': source_model_list,
                'point_source_model_list': point_source_list
                 }

# display the initial simulated image
cmap_string = 'viridis'
cmap = plt.get_cmap(cmap_string)
cmap.set_bad(color='k', alpha=1.)
cmap.set_under('k')

v_min = -4
v_max = 2

f, axes = plt.subplots(1, 1, figsize=(6, 6), sharex=False, sharey=False)
ax = axes
im = ax.matshow(np.log10(image_sim), origin='lower', vmin=v_min, vmax=v_max, cmap=cmap, extent=[0, 1, 0, 1])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.autoscale(False)
plt.show()

#%%
















