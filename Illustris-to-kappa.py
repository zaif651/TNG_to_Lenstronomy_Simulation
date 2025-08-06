#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 00:51:04 2025

@author: abdullahalzaif
"""

#ILLUSTRIS_TNG

#%%

import os
import requests
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Hubble parameter from TNG cosmology
h = 0.6774

#%%
# API configuration
API_KEY = os.getenv("TNG_API_KEY")
HEADERS = {"api-key": "4d3728701fffb127589a0a0fe475db79"}

def get(url, params=None):
    """
    Helper function to perform GET requests from the IllustrisTNG API.
    If a JSON response is returned, parse and return it as a dict.
    If a file is returned (has a 'content-disposition' header),
    save it locally and return the filename.
    Otherwise, return the raw requests.Response object.
    """
    # make HTTP GET request to path
    response = requests.get(url, params=params, headers=HEADERS)

    # raise exception if response code is not HTTP SUCCESS (200)
    response.raise_for_status()  # raise an exception for HTTP errors

    # If the response is JSON, parse and return as a dict:
    if response.headers.get("content-type") == "application/json":
        return response.json()   # parse json responses automatically

    # If this is a file download, save to disk and return the local filename
    if "content-disposition" in response.headers:
        # e.g. 'attachment; filename="cutout_12345.hdf5"'
        cd = response.headers["content-disposition"]
        # Extract the actual filename:
        filename = cd.split("filename=")[1].strip(';"')
        with open(filename, "wb") as f:
            f.write(response.content)
        return filename

    # Otherwise just return the raw response:
    return response

#%%

baseUrl = 'http://www.tng-project.org/api/'
response = get(baseUrl)
print(response.keys(), len(response['simulations']))

#%%

print(response['simulations'][0])
names = [sim['name'] for sim in response['simulations']]
for i in names :
  print(i)
  
#%%  Selecting Subhalos within a Mass Range, and storing stellar, gas and DM mass and sSFR
"""
Key : \\
Halo $-$ Galaxy Cluster \\
Subhalo $-$ individual galaxy within cluster """

# Redshift of interest
z_snap = 0.5

# for Illustris-1 at 𝑧=0, search for all subhalos with total mass mass_min Solar<𝑀<mass_max Solar
# print the number returned, and the Subfind IDs of the first five results

base_url = "http://www.tng-project.org/api/Illustris-1/"

# Snapshot-to-redshift mapping (not needed, just in case)
snapshots_url = base_url+"/snapshots/"
snapshots_data = get(snapshots_url)
snap_to_z = {snap['number']: snap['redshift'] for snap in snapshots_data}

# first convert log solar masses into group catalog units
mass_min = 10**12.5 / 1e10 * 0.704
mass_max = 10**13. / 1e10 * 0.704

# form the search_query string by hand for once
search_query = "?mass__gt=" + str(mass_min) + "&mass__lt=" + str(mass_max)

# form the url and make the request
url = base_url+f"snapshots/z={z_snap}/subhalos/" + search_query
print(f"URL constructed: {url}")

# Set the limit
subhalos = get(url, {'limit':1000})
print('Number of sub-halos satisfying mass criterion:', subhalos['count'])

nids = 300
ids = [subhalos['results'][i]['id'] for i in range(min(nids, len(subhalos['results'])))]

print(f"First 10 IDs: {ids[0:10]}")

# Create a list to store the sSFR results for each subhalo
ssfr_data = []

# Loop through the subhalo list to compute sSFR for each subhalo
counter = 0
for sub in subhalos['results']:
    # Retrieve detailed info for the subhalo
    sub_detail = get(sub['url'])

    # Extract star formation rate and stellar mass etc
    sfr = sub_detail.get('sfr', None)
    mass_stars = sub_detail.get('mass_stars', None)
    mass_gas = sub_detail.get('mass_gas', None)
    mass_dm = sub_detail.get('mass_dm', None)

    # Check if mass_stars is valid (non-zero) to avoid division by zero
    if sfr is not None and mass_stars is not None and mass_stars > 0:
        sSFR = sfr / mass_stars
    else:
        sSFR = 0

    print(f"{counter} Subhalo ID {sub['id']}: SFR = {sfr}, Stellar Mass = {mass_stars}, Gas Mass = {mass_gas}, sSFR = {sSFR}")

     # Store the subhalo's id and computed sSFR (along with sfr and mass_stars for reference)
    ssfr_data.append({
        'id': sub['id'],
        'sfr': sfr,
        'mass_stars': mass_stars,
        'mass_gas': mass_gas,
        'mass_dm': mass_dm,
        'sSFR': sSFR
    })
    counter += 1

#%% Show Plots of selected subhalo properties

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Extract sSFR and stellar mass values from ssfr_data
stellar_mass = [entry['mass_stars'] for entry in ssfr_data]

dm_mass = [entry['mass_dm'] for entry in ssfr_data]
ssfr_values = [entry['sSFR'] for entry in ssfr_data]

# Create a figure with two subplots of equal width
fig, axs = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1]})

# Set the x-axis limits for both subplots
axs[0].set_xlim(2e10, 2e12)
axs[1].set_xlim(2e10, 2e12)

# First subplot: sSFR vs Stellar Mass color-coded by Dark Matter Mass
sc0 = axs[0].scatter(1e+10*np.array(stellar_mass), ssfr_values, c=dm_mass, cmap='viridis', alpha=0.7)
axs[0].set_xlabel('Stellar Mass [$10^{10} M_{\odot}$]')
axs[0].set_ylabel('sSFR [yr$^{-1}$]')
axs[0].set_title('sSFR vs Stellar Mass (Color-coded by DM Mass)')
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].grid(True)

# Add a colorbar inside the right subplot without affecting its width
cax = inset_axes(axs[0], width="3%", height="80%", loc='upper right', borderpad=4.5)
cbar = fig.colorbar(sc0, cax=cax)
cbar.set_label('DM Mass [$10^{10} M_{\odot}$]')

# Extract gas mass values from ssfr_data
gas_mass = [entry['mass_gas'] for entry in ssfr_data]

# Second subplot: sSFR vs Stellar Mass color-coded by Gas Mass
sc1 = axs[1].scatter(1e+10*np.array(stellar_mass), ssfr_values, c=gas_mass, cmap='plasma', alpha=0.7)
axs[1].set_xlabel('Stellar Mass [$10^{10} M_{\odot}$]')
axs[1].set_ylabel('sSFR [yr$^{-1}$]')
axs[1].set_title('sSFR vs Stellar Mass (Color-coded by Gas Mass)')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid(True)

# Add a colorbar inside the right subplot without affecting its width
cax = inset_axes(axs[1], width="3%", height="80%", loc='upper right', borderpad=4.5)
cbar = fig.colorbar(sc1, cax=cax)
cbar.set_label('Gas Mass [$10^{10} M_{\odot}$]')

plt.tight_layout()
plt.show()

#%%

print(type(dm_mass))

#%%

# Set the sSFR threshold
ssfr_threshold = 1e-2

# Set the stellar mass threshold
stellar_mass_threshold = 3e+11

# Find the indices of subhalos that satisfy the sSFR threshold
selected_indices = [i for i, entry in enumerate(ssfr_data) if (entry['sSFR'] < ssfr_threshold) & (1e+10*entry['mass_stars'] > stellar_mass_threshold)]
print(selected_indices)

# Request the subhalo details and snapshot cutout for stars
id_num = 0
id_num = selected_indices[id_num]

subhalo_url = base_url + f"snapshots/z={z_snap}/subhalos/" + str(ids[id_num]) + "/"
subhalo = get(subhalo_url)

print('Sub-halo center:', subhalo['pos_x'], subhalo['pos_y'], subhalo['pos_z'])
print('Sub-halo velocity:', subhalo['vel_x'], subhalo['vel_y'], subhalo['vel_z'])

# Request star particle data (Coordinates, Velocities, Masses)
cutout_url = subhalo_url + "cutout.hdf5"

print("Downloading sub-halo cutout file...")
cutout_request = {'dm': 'Coordinates,Velocities',
    'stars': 'Coordinates,Velocities,Masses,GFM_StellarFormationTime,GFM_Metallicity'}
cutout = get(cutout_url, cutout_request)

# To track where the subhalo’s stars originated,
# get the main progenitor branch (MPB) from the merger tree:
#tree_url = subhalo['trees']['sublink_mpb']  # Main progenitor branch
#tree_file = get(tree_url)

# Load the merger tree
#print("Loading the merger tree...")
#with h5py.File(tree_file, 'r') as f:
#    progenitor_ids = f['SubhaloID'][:]  # IDs of subhalos at each snapshot
#    progenitor_snapshots = f['SnapNum'][:]  # Snapshots at which these subhalos existed

# Open HDF5 cutout and extract data
print("Reading the snapshot file...")
with h5py.File(cutout, 'r') as f:
    # DM
    # Positions (X, Y, Z)
    dm_positions = f['PartType1']['Coordinates'][:]

    # Velocities (VX, VY, VZ)
    dm_vels = f['PartType1']['Velocities'][:]

    # STARS
    # Positions (X, Y, Z)
    star_positions = f['PartType4']['Coordinates'][:]

    # Velocities (VX, VY, VZ)
    star_vels = f['PartType4']['Velocities'][:]

    # Convert masses to Solar masses
    masses = f['PartType4']['Masses'][:] * (1e10 / h)

    # Formation times
    formation_times = f['PartType4']['GFM_StellarFormationTime'][:]

    # Metal
    metal = f['PartType4']['GFM_Metallicity'][:]

print("Done.")

#%%

# 2D HIST setup
bins_in = [300, 300]

bins_in2 = [500, 500]

xr_in1 = np.array([-25, 25])
yr_in1 = np.array([-25, 25])

bins = [75, 75]
xr2 = np.array([-100, 100])
yr2 = np.array([-100, 100])

# X,Y,Z
x = star_positions[:, 0]
y = star_positions[:, 1]
z = star_positions[:, 2]

# VX,VY,VZ
vx =  star_vels[:, 0]
vy =  star_vels[:, 1]
vz =  star_vels[:, 2]

# Compute offsets relative to the **subhalo center** (not parent halo!)
x_off = x - subhalo['pos_x']
y_off = y - subhalo['pos_y']
z_off = z - subhalo['pos_z']

vx_off = vx - subhalo['vel_x']
vy_off = vy - subhalo['vel_y']
vz_off = vz - subhalo['vel_z']

#%%

x_min, x_max = np.min(x_off), np.max(x_off)
y_min, y_max = np.min(y_off), np.max(y_off)

xr_in3 = np.array([x_min, x_max])
yr_in3 = np.array([y_min,y_max])

x_extent = [-300, 300]
y_extent = [-300, 300]

#%%


# Create 2D histogram in X-Y offset
hist, xedges, yedges = np.histogram2d(x_off, y_off, range = [x_extent, y_extent], bins=[400, 400], weights=masses)

#%%
# Mask zero values before log transformation
w0 = hist > 0

# Apply log10 transformation safely

#hist[w0] = np.log10(hist[w0])


# Compute vmin and vmax using the 5th and 99.99th percentiles
vmin, vmax = np.percentile(hist[w0], [5, 99.99])

# Plot the histogram
plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
           vmin=vmin, vmax=vmax, aspect='auto', cmap='magma')

plt.colorbar(label=r'$\log M_{\odot}$')
plt.xlabel(r'$\Delta x_{\rm offset}$ [ckpc/h]')
plt.ylabel(r'$\Delta y_{\rm offset}$ [ckpc/h]')
plt.title('Stellar density in native CS relative to progenitor')
plt.show()


#%%
# Calculate the area of each grid cell (delta_x * delta_y)
dx = xedges[1] - xedges[0]
dy = yedges[1] - yedges[0]

# Surface density (number of particles per unit area)
surface_density = hist / (dx * dy)

p0 = surface_density > 0

surface_density[p0]= surface_density[p0]

#surface_density_log = surface_density

#surface_density_log[p0] = np.log10(surface_density_log[p0])

#%%


plt.figure(figsize=(8, 6))
plt.imshow(surface_density.T, origin='lower', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
plt.colorbar(label="$\log_{10}$(Surface Density)")
plt.xlabel('X Position (kpc)')
plt.ylabel('Y Position (kpc)')
plt.title('2D Projected Surface Density')
plt.show()


#%%

x = surface_density

surface_density_log = surface_density

surface_density_log[p0] = np.log10(x[p0])

plt.figure(figsize=(8, 6))
plt.imshow(surface_density_log.T, origin='lower', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
plt.colorbar(label="$\log_{10}$(Surface Density)")
plt.xlabel('X Position (kpc)')
plt.ylabel('Y Position (kpc)')
plt.title('2D Projected Surface Density')
plt.show()

#%%
from scipy.ndimage import gaussian_filter

# Smooth the surface density
smoothed_density = gaussian_filter(surface_density, sigma=1)  # sigma controls the smoothing level

# Plot the smoothed surface density
plt.figure(figsize=(8, 6))
plt.imshow(smoothed_density.T, origin='lower', aspect='auto', extent=[x_min, x_max, y_min, y_max], cmap='viridis')
plt.colorbar(label="$\log_{10}$(Surface Density)")
plt.xlabel('X Position (kpc)')
plt.ylabel('Y Position (kpc)')
plt.title('Smoothed 2D Projected Surface Density')
plt.show()


#%% 2D array surface mass density of the galaxy

surface_mass = hist / (dx * dy)

p0 = surface_mass > 0

surface_mass[p0]= surface_mass[p0]


#%% Define a lens configuration and compute critical surface density

from astropy.cosmology import Planck15 as cosmo
from astropy.constants import G, c
import astropy.units as u


z_lens = 0.5
z_source = 1.5

D_d = cosmo.angular_diameter_distance(z_lens)
D_s = cosmo.angular_diameter_distance(z_source)
D_ds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source)

sigma_crit = (c**2 / (4 * np.pi * G)) * (D_s / (D_d * D_ds))
sigma_crit = sigma_crit.to(u.Msun / u.pc**2).value  # Msun / pc^2


#%%

#surface_mass_smoothed = gaussian_filter(surface_mass, sigma=1)  # sigma controls the smoothing level

#surface_mass_smoothed[p0]= surface_mass_smoothed[p0] * 10**(-6)

#%%

kappa_observed = surface_mass / sigma_crit

p0 = kappa_observed > 0

kappa_observed[p0] = np.log10(kappa_observed[p0])

plt.imshow((kappa_observed.T), origin='lower', cmap='inferno', extent=[-300, 300, -300, 300])
plt.colorbar(label='Convergence κ')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title('Projected Convergence Map from Stars')
plt.show()

#%%

np.save("/Users/abdullahalzaif/Desktop/PIII_Project_Strong_Lensing/IllustrisTNG/kappa_observed.npy", kappa_observed, allow_pickle=True)
np.save("/Users/abdullahalzaif/Desktop/PIII_Project_Strong_Lensing/IllustrisTNG/x_coord.npy", xedges, allow_pickle=True)
np.save("/Users/abdullahalzaif/Desktop/PIII_Project_Strong_Lensing/IllustrisTNG/y_coord.npy", yedges, allow_pickle=True)


#%%
"""
#%%

radii = np.sqrt(x_off**2 + y_off**2)
angles = np.arctan2(y_off, x_off)+np.pi

bin_polar = [300, 300]

r_in = [0, np.max(radii)]
angle_in = [0, 2*np.pi]

#%%
# Create 2D histogram in Polar coord
hist_polar, r_edges, ang_edges = np.histogram2d(radii, angles, range = [r_in,angle_in], bins=bin_polar, weights=masses)

# Mask zero values before log transformation
w0 = hist_polar > 0

# Apply log10 transformation safely

hist_polar[w0] = np.log10(hist_polar[w0])


# Compute vmin and vmax using the 5th and 99.99th percentiles
vmin, vmax = np.percentile(hist_polar[w0], [5, 99.99])

# Plot the histogram
plt.imshow(hist_polar.T, origin='lower', extent=[r_edges[0], r_edges[-1], ang_edges[0], ang_edges[-1]],
           vmin=vmin, vmax=vmax, aspect='auto', cmap='magma')

plt.colorbar(label=r'$\log M_{\odot}$')
plt.xlabel(r'$radius from subhalo centre, r$ [ckpc/h]')
plt.ylabel(r'$angle$ [radians]')
plt.title('Stellar density in native CS relative to progenitor')
plt.show()

#%%

plt.figure()
plt.plot(radii, masses, '.')
plt.show()
#%%

hist1D, redges = np.histogram(radii, range = r_in, bins=300, weights=masses)

w0 = hist1D > 0

# Apply log10 transformation safely

#hist[w0] = np.log10(hist[w0])


#%%

plt.figure()

plt.hist(radii, range = r_in, bins=500, weights=masses)

plt.show()

#%%

r_ = redges[1:] - 0.5138383765

plt.plot(r_, hist1D, '.')

"""











