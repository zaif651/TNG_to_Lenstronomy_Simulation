#!/usr/bin/env python
"""
Modified debug script for IllustrisTNG + lenstronomy INTERPOL
This version auto-calculates the second derivatives
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Import the debug functions
sys.path.append('/tmp')
from debug_interpol import (debug_interpol_setup, 
                           test_simple_lensing_simulation, 
                           suggest_fixes)

# Load variables from your notebook
# Replace these with the actual file paths to your data
# For now, I'm assuming you have these arrays saved as .npy files
try:
    # Try to import from your notebook environment
    from IPython import get_ipython
    
    # If running in Jupyter notebook
    if get_ipython() is not None:
        print("Running in Jupyter - trying to access notebook variables...")
        
        # Access variables directly from notebook namespace
        grid_interp_x = x_2d_arcsec  # The x grid in arcsec
        grid_interp_y = y_2d_arcsec  # The y grid in arcsec
        f_ = lensing_potential      # The lensing potential
        f_x = alpha_x_custom        # x deflection angle
        f_y = alpha_y_custom        # y deflection angle
        
        print("Successfully loaded variables from notebook")
        
    else:
        # If running as standalone script, try to load from files
        raise ImportError("Not running in notebook")
        
except (ImportError, NameError):
    # Standalone mode - load from files (adjust paths as needed)
    print("Running in standalone mode - loading from files...")
    
    data_dir = input("Enter the directory path containing your data files: ")
    
    grid_interp_x = np.load(os.path.join(data_dir, 'x_grid_arcsec.npy'))
    grid_interp_y = np.load(os.path.join(data_dir, 'y_grid_arcsec.npy'))
    f_ = np.load(os.path.join(data_dir, 'lensing_potential.npy'))
    f_x = np.load(os.path.join(data_dir, 'alpha_x.npy'))
    f_y = np.load(os.path.join(data_dir, 'alpha_y.npy'))

# Calculate pixel scale from grid
if len(grid_interp_x.shape) > 1:
    pixel_scale = np.mean(np.diff(grid_interp_x[0, :]))
else:
    # If 1D array, reshape to 2D
    nx = int(np.sqrt(len(grid_interp_x)))
    grid_interp_x = grid_interp_x.reshape(nx, nx)
    grid_interp_y = grid_interp_y.reshape(nx, nx)
    f_ = f_.reshape(nx, nx)
    f_x = f_x.reshape(nx, nx)
    f_y = f_y.reshape(nx, nx)
    pixel_scale = np.mean(np.diff(grid_interp_x[0, :]))

# Print basic info about the data
print(f"\nInput data shapes:")
print(f"x_grid: {grid_interp_x.shape}")
print(f"y_grid: {grid_interp_y.shape}")
print(f"potential: {f_.shape}")
print(f"alpha_x: {f_x.shape}")
print(f"alpha_y: {f_y.shape}")
print(f"Pixel scale: {pixel_scale:.6f} arcsec")

# Calculate second derivatives (Hessian)
print("\nCalculating second derivatives...")

# Calculate second derivatives using gradient
# Note: gradient of deflection = gradient of -gradient of potential
# So we negate the result to get the correct Hessian
# Note that this assumes you're using the right pixel scale in arcsec
f_xx = -np.gradient(f_x, pixel_scale, axis=1)  # d^2f/dx^2
f_yy = -np.gradient(f_y, pixel_scale, axis=0)  # d^2f/dy^2

# For mixed derivative, we have options:
# Option 1: Use cross-derivative of first derivatives
f_xy_1 = -np.gradient(f_x, pixel_scale, axis=0)  # d/dy of df/dx
f_xy_2 = -np.gradient(f_y, pixel_scale, axis=1)  # d/dx of df/dy

# The mixed derivatives should be equal (up to numerical errors),
# so we take the average for better accuracy
f_xy = 0.5 * (f_xy_1 + f_xy_2)

print("Second derivatives calculated successfully")

# Check if the potential needs scaling
max_potential = np.abs(f_).max()
if max_potential < 1e-8:
    print("\n⚠️  WARNING: Potential values are very small. Testing scaled versions...")
    
    # Create scaled versions for testing
    scaling_factors = [10, 100, 1000, 10000]
    for factor in scaling_factors:
        print(f"\n--- Testing with scaling factor {factor} ---")
        
        # Create scaled versions
        f_scaled = f_ * factor
        f_x_scaled = f_x * factor 
        f_y_scaled = f_y * factor
        f_xx_scaled = f_xx * factor
        f_yy_scaled = f_yy * factor
        f_xy_scaled = f_xy * factor
        
        # Run diagnostics on scaled version
        success, lens_model, kwargs_lens = debug_interpol_setup(
            grid_interp_x, grid_interp_y, 
            f_scaled, f_x_scaled, f_y_scaled, 
            f_xx_scaled, f_yy_scaled, f_xy_scaled
        )
        
        if success:
            print(f"\nTrying simulation with scaling factor {factor}...")
            test_simple_lensing_simulation(lens_model, kwargs_lens, 
                                         grid_interp_x, grid_interp_y)
            
            # If we've found a working scaling factor, stop testing
            choice = input(f"\nDid this scaling factor ({factor}) work well? (y/n): ").strip().lower()
            if choice == 'y':
                print(f"\n✓ Success! Using scaling factor {factor}")
                print(f"👉 RECOMMENDATION: Scale your lensing potential by {factor} in your notebook")
                
                # Update for suggestions
                f_ = f_scaled
                f_x = f_x_scaled
                f_y = f_y_scaled
                break
else:
    # Run diagnostics on original data
    print("\n--- Running diagnostics on original data ---")
    success, lens_model, kwargs_lens = debug_interpol_setup(
        grid_interp_x, grid_interp_y, 
        f_, f_x, f_y, 
        f_xx, f_yy, f_xy
    )
    
    if success:
        print("\nTrying simulation with original data...")
        test_simple_lensing_simulation(lens_model, kwargs_lens, grid_interp_x, grid_interp_y)

# Provide suggested fixes
print("\n--- Final Analysis ---")
suggest_fixes(grid_interp_x, grid_interp_y, f_, f_x, f_y)

# Create diagnostic plots
print("\nCreating diagnostic plots...")

plt.figure(figsize=(16, 12))

# Plot lensing potential
plt.subplot(231)
plt.imshow(f_, origin='lower', cmap='viridis')
plt.colorbar(label='Lensing Potential')
plt.title('Lensing Potential')

# Plot deflection angles
plt.subplot(232)
plt.imshow(f_x, origin='lower', cmap='RdBu')
plt.colorbar(label='Deflection X (arcsec)')
plt.title('Deflection Angle X')

plt.subplot(233)
plt.imshow(f_y, origin='lower', cmap='RdBu')
plt.colorbar(label='Deflection Y (arcsec)')
plt.title('Deflection Angle Y')

# Plot convergence
plt.subplot(234)
kappa = 0.5 * (f_xx + f_yy)
plt.imshow(kappa, origin='lower', cmap='viridis')
plt.colorbar(label='Convergence κ')
plt.title('Convergence (κ)')

# Plot deflection field
plt.subplot(235)
step = max(1, grid_interp_x.shape[0] // 20)  # Downsample for clarity
plt.quiver(grid_interp_x[::step, ::step], 
          grid_interp_y[::step, ::step],
          f_x[::step, ::step], 
          f_y[::step, ::step],
          scale=1.0, alpha=0.8)
plt.axis('equal')
plt.title('Deflection Field')

# Plot deflection magnitude
plt.subplot(236)
defl_mag = np.sqrt(f_x**2 + f_y**2)
plt.imshow(defl_mag, origin='lower', cmap='viridis')
plt.colorbar(label='Deflection Magnitude (arcsec)')
plt.title('Deflection Magnitude')

plt.tight_layout()
plt.savefig('interpol_diagnostics.png', dpi=150)
plt.show()

print("\nDiagnostic complete! See the plots for visualization.")
print("If you want to make the lensing effect stronger, try:")
print("1. Scale your lensing potential by a factor (e.g., 100-1000)")
print("2. Check the sign of your deflection angles (should be -∇ψ)")
print("3. Ensure your coordinate grids are in arcsec and properly centered")
print("4. Try using a brighter, more compact source")
