"""
Debug script for INTERPOL lens model integration with IllustrisTNG data
This script helps diagnose why INTERPOL isn't producing lensing images
"""

import numpy as np
import matplotlib.pyplot as plt
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

def debug_interpol_setup(grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy):
    """
    Debug function to check INTERPOL setup and identify potential issues
    
    Parameters:
    -----------
    All parameters should be from your IllustrisTNG INTERPOL setup
    """
    print("=== INTERPOL Debug Analysis ===\n")
    
    # 1. Check grid properties
    print("1. Grid Properties:")
    print(f"   Grid shape: {grid_interp_x.shape}")
    print(f"   X range: [{grid_interp_x.min():.3f}, {grid_interp_x.max():.3f}] arcsec")
    print(f"   Y range: [{grid_interp_y.min():.3f}, {grid_interp_y.max():.3f}] arcsec")
    print(f"   Grid spacing: {np.diff(grid_interp_x[0, :]).mean():.3f} arcsec")
    
    # 2. Check potential values
    print("\n2. Lensing Potential (f_):")
    print(f"   Min: {f_.min():.6e}")
    print(f"   Max: {f_.max():.6e}")
    print(f"   Mean: {f_.mean():.6e}")
    print(f"   Std: {f_.std():.6e}")
    
    # Check if potential is too small (common issue)
    if np.abs(f_).max() < 1e-10:
        print("   ⚠️  WARNING: Potential values are very small - may not produce detectable lensing")
    
    # 3. Check deflection angles
    print("\n3. Deflection Angles:")
    print(f"   f_x range: [{f_x.min():.6e}, {f_x.max():.6e}]")
    print(f"   f_y range: [{f_y.min():.6e}, {f_y.max():.6e}]")
    print(f"   Max deflection magnitude: {np.sqrt(f_x**2 + f_y**2).max():.6e} arcsec")
    
    # 4. Check second derivatives (Hessian)
    print("\n4. Hessian Components:")
    print(f"   f_xx range: [{f_xx.min():.6e}, {f_xx.max():.6e}]")
    print(f"   f_yy range: [{f_yy.min():.6e}, {f_yy.max():.6e}]")
    print(f"   f_xy range: [{f_xy.min():.6e}, {f_xy.max():.6e}]")
    
    # 5. Convergence and shear analysis
    kappa = 0.5 * (f_xx + f_yy)  # Convergence
    gamma1 = 0.5 * (f_xx - f_yy)  # Shear component 1
    gamma2 = f_xy                  # Shear component 2
    
    print("\n5. Lensing Properties:")
    print(f"   Convergence κ range: [{kappa.min():.6e}, {kappa.max():.6e}]")
    print(f"   Max |κ|: {np.abs(kappa).max():.6e}")
    print(f"   Shear γ₁ range: [{gamma1.min():.6e}, {gamma1.max():.6e}]")
    print(f"   Shear γ₂ range: [{gamma2.min():.6e}, {gamma2.max():.6e}]")
    print(f"   Max shear magnitude: {np.sqrt(gamma1**2 + gamma2**2).max():.6e}")
    
    # Critical convergence check
    if np.abs(kappa).max() < 0.01:
        print("   ⚠️  WARNING: Very low convergence - lensing effects may be negligible")
    if np.abs(kappa).max() > 1.0:
        print("   ⚠️  NOTE: High convergence detected - strong lensing regime")
    
    # 6. Test basic INTERPOL functionality
    print("\n6. Testing INTERPOL Model:")
    try:
        lens_model_list = ['INTERPOL']
        lens_model = LensModel(lens_model_list)
        
        kwargs_lens = [{'grid_interp_x': grid_interp_x,
                       'grid_interp_y': grid_interp_y,
                       'f_': f_,
                       'f_x': f_x,
                       'f_y': f_y,
                       'f_xx': f_xx,
                       'f_yy': f_yy,
                       'f_xy': f_xy}]
        
        # Test evaluation at a few points
        test_x = np.array([0.0, 1.0, -1.0])
        test_y = np.array([0.0, 1.0, -1.0])
        
        alpha_x, alpha_y = lens_model.alpha(test_x, test_y, kwargs_lens)
        print(f"   ✓ INTERPOL model created successfully")
        print(f"   ✓ Deflection at (0,0): ({alpha_x[0]:.6e}, {alpha_y[0]:.6e})")
        print(f"   ✓ Max test deflection: {np.sqrt(alpha_x**2 + alpha_y**2).max():.6e}")
        
        return True, lens_model, kwargs_lens
        
    except Exception as e:
        print(f"   ❌ ERROR: Failed to create INTERPOL model: {e}")
        return False, None, None

def test_simple_lensing_simulation(lens_model, kwargs_lens, grid_interp_x, grid_interp_y):
    """
    Test if the INTERPOL model can produce lensing images
    """
    print("\n7. Simple Lensing Simulation Test:")
    
    try:
        # Create a simple setup
        nx, ny = 100, 100
        pixel_scale = 0.1  # arcsec/pixel
        
        # Use coordinate range similar to your INTERPOL grid
        x_center = (grid_interp_x.max() + grid_interp_x.min()) / 2
        y_center = (grid_interp_y.max() + grid_interp_y.min()) / 2
        
        # Set up data class
        kwargs_data = {
            'nx': nx, 'ny': ny,
            'pixel_scale': pixel_scale,
            'ra_at_xy0': x_center, 'dec_at_xy0': y_center
        }
        data_class = ImageData(**kwargs_data)
        
        # Simple PSF
        psf_model = PSF(psf_type='GAUSSIAN', fwhm=0.1, pixel_scale=pixel_scale)
        
        # Create a simple source (Gaussian)
        light_model_list = ['GAUSSIAN']
        light_model = LightModel(light_model_list)
        kwargs_source = [{'amp': 1000, 'sigma': 0.5, 'center_x': 0.2, 'center_y': 0.1}]
        
        # No point sources for this test
        point_source_model = PointSource([])
        
        # Create image model
        image_model = ImageModel(data_class, psf_model, lens_model, [], 
                               light_model, [], point_source_model)
        
        # Generate lensed image
        image = image_model.image(kwargs_lens, [], kwargs_source, [], [])
        
        # Check if lensing occurred
        unlensed_image = light_model.surface_brightness(data_class.pixel_coordinates[0], 
                                                      data_class.pixel_coordinates[1], 
                                                      kwargs_source)
        
        max_lensed = np.max(image)
        max_unlensed = np.max(unlensed_image)
        
        print(f"   ✓ Simulation completed successfully")
        print(f"   ✓ Unlensed image max: {max_unlensed:.2e}")
        print(f"   ✓ Lensed image max: {max_lensed:.2e}")
        print(f"   ✓ Image flux ratio: {max_lensed/max_unlensed:.3f}")
        
        if np.abs(max_lensed - max_unlensed) / max_unlensed < 0.01:
            print("   ⚠️  WARNING: Very small difference between lensed/unlensed - weak lensing")
        else:
            print("   ✓ Significant lensing detected!")
            
        return True, image, unlensed_image
        
    except Exception as e:
        print(f"   ❌ ERROR: Simulation failed: {e}")
        import traceback
        print(f"   Full traceback:\n{traceback.format_exc()}")
        return False, None, None

def suggest_fixes(grid_interp_x, grid_interp_y, f_, f_x, f_y):
    """
    Suggest potential fixes based on the analysis
    """
    print("\n=== SUGGESTED FIXES ===")
    
    # Check potential magnitude
    if np.abs(f_).max() < 1e-8:
        print("1. ⚠️  Lensing potential is very weak:")
        print("   - Check if your IllustrisTNG mass is in correct units")
        print("   - Verify lensing potential calculation (should be dimensionless)")
        print("   - Consider scaling up the potential artificially for testing")
    
    # Check deflection magnitude
    max_defl = np.sqrt(f_x**2 + f_y**2).max()
    if max_defl < 1e-6:
        print("2. ⚠️  Deflection angles are very small:")
        print("   - Check gradient calculation method")
        print("   - Verify pixel size scaling in gradient computation")
        print("   - Ensure proper coordinate system conversion")
    
    # Check grid coverage
    grid_size = np.abs(grid_interp_x.max() - grid_interp_x.min())
    if grid_size < 2.0:
        print("3. ⚠️  Small grid coverage:")
        print("   - Consider expanding coordinate range")
        print("   - Ensure grid covers region where sources will be placed")
    
    # Check grid resolution
    spacing = np.diff(grid_interp_x[0, :]).mean()
    if spacing > 0.5:
        print("4. ⚠️  Coarse grid resolution:")
        print("   - Consider increasing grid resolution")
        print("   - Finer sampling may reveal lensing features")
    
    print("\n=== DEBUGGING STEPS ===")
    print("1. Scale test: Multiply f_, f_x, f_y by factor of 10-100 and test")
    print("2. Check units: Ensure all quantities are in proper lenstronomy units")
    print("3. Verify calculation: Double-check lensing potential formula")
    print("4. Test with simple lens: Compare with analytical SIE model")
    print("5. Visualize: Plot 2D maps of potential, deflections, convergence")

# Example usage (you would call this with your actual data):
"""
# Load your IllustrisTNG INTERPOL parameters
grid_interp_x = your_grid_x
grid_interp_y = your_grid_y  
f_ = your_lensing_potential
f_x = your_deflection_x
f_y = your_deflection_y
f_xx = your_second_derivative_xx
f_yy = your_second_derivative_yy
f_xy = your_second_derivative_xy

# Run diagnostics
success, lens_model, kwargs_lens = debug_interpol_setup(
    grid_interp_x, grid_interp_y, f_, f_x, f_y, f_xx, f_yy, f_xy
)

if success:
    test_simple_lensing_simulation(lens_model, kwargs_lens, grid_interp_x, grid_interp_y)

suggest_fixes(grid_interp_x, grid_interp_y, f_, f_x, f_y)
"""

print("INTERPOL debugging script created successfully!")
print("Copy this script and run it with your IllustrisTNG data to diagnose issues.")
