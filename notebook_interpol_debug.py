"""
Simplified debug script for INTERPOL - Paste this into your Jupyter notebook
"""
import numpy as np

def debug_interpol_with_auto_second_derivatives(x_grid, y_grid, potential, alpha_x, alpha_y):
    """
    Run diagnostics on INTERPOL lens model setup, calculating second derivatives automatically
    
    Parameters:
    -----------
    x_grid: 2D array of x coordinates (arcsec)
    y_grid: 2D array of y coordinates (arcsec) 
    potential: 2D array of lensing potential (dimensionless)
    alpha_x: 2D array of x deflection angles (arcsec)
    alpha_y: 2D array of y deflection angles (arcsec)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.ImSim.image_model import ImageModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.PointSource.point_source import PointSource
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    
    # Calculate pixel scale from grid
    pixel_scale = np.mean(np.diff(x_grid[0, :]))
    
    print("=== INTERPOL Debug Analysis ===\n")
    
    # 1. Check grid properties
    print("1. Grid Properties:")
    print(f"   Grid shape: {x_grid.shape}")
    print(f"   X range: [{x_grid.min():.3f}, {x_grid.max():.3f}] arcsec")
    print(f"   Y range: [{y_grid.min():.3f}, {y_grid.max():.3f}] arcsec")
    print(f"   Grid spacing: {pixel_scale:.3f} arcsec")
    
    # 2. Check potential values
    print("\n2. Lensing Potential:")
    print(f"   Min: {potential.min():.6e}")
    print(f"   Max: {potential.max():.6e}")
    print(f"   Mean: {potential.mean():.6e}")
    print(f"   Std: {potential.std():.6e}")
    
    # Check if potential is too small (common issue)
    if np.abs(potential).max() < 1e-8:
        print("   ⚠️  WARNING: Potential values are very small - may not produce detectable lensing")
    
    # 3. Check deflection angles
    print("\n3. Deflection Angles:")
    print(f"   alpha_x range: [{alpha_x.min():.6e}, {alpha_x.max():.6e}]")
    print(f"   alpha_y range: [{alpha_y.min():.6e}, {alpha_y.max():.6e}]")
    print(f"   Max deflection magnitude: {np.sqrt(alpha_x**2 + alpha_y**2).max():.6e} arcsec")
    
    # Calculate second derivatives
    print("\n4. Calculating second derivatives...")
    
    # Calculate second derivatives using gradient of deflections
    f_xx = -np.gradient(alpha_x, pixel_scale, axis=1)  # d^2f/dx^2
    f_yy = -np.gradient(alpha_y, pixel_scale, axis=0)  # d^2f/dy^2
    
    # For mixed derivative, average the two possible calculations
    f_xy_1 = -np.gradient(alpha_x, pixel_scale, axis=0)  # d/dy of df/dx
    f_xy_2 = -np.gradient(alpha_y, pixel_scale, axis=1)  # d/dx of df/dy
    f_xy = 0.5 * (f_xy_1 + f_xy_2)
    
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
        
        kwargs_lens = [{'grid_interp_x': x_grid,
                       'grid_interp_y': y_grid,
                       'f_': potential,
                       'f_x': alpha_x,
                       'f_y': alpha_y,
                       'f_xx': f_xx,
                       'f_yy': f_yy,
                       'f_xy': f_xy}]
        
        # Test evaluation at a few points
        test_x = np.array([0.0, 1.0, -1.0])
        test_y = np.array([0.0, 1.0, -1.0])
        
        alpha_x_test, alpha_y_test = lens_model.alpha(test_x, test_y, kwargs_lens)
        print(f"   ✓ INTERPOL model created successfully")
        print(f"   ✓ Deflection at (0,0): ({alpha_x_test[0]:.6e}, {alpha_y_test[0]:.6e})")
        print(f"   ✓ Max test deflection: {np.sqrt(alpha_x_test**2 + alpha_y_test**2).max():.6e}")
        
        # 7. Test simulation
        print("\n7. Simple Lensing Simulation Test:")
        
        # Create a simple setup
        nx, ny = 100, 100
        pixel_scale_sim = 0.1  # arcsec/pixel
        
        # Use coordinate range similar to INTERPOL grid
        x_center = (x_grid.max() + x_grid.min()) / 2
        y_center = (y_grid.max() + y_grid.min()) / 2
        
        # Set up data class
        kwargs_data = {
            'nx': nx, 'ny': ny,
            'pixel_scale': pixel_scale_sim,
            'ra_at_xy0': x_center, 'dec_at_xy0': y_center
        }
        data_class = ImageData(**kwargs_data)
        
        # Simple PSF
        psf_model = PSF(psf_type='GAUSSIAN', fwhm=0.1, pixel_scale=pixel_scale_sim)
        
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
        
        # Create visualization of results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.imshow(unlensed_image, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Unlensed Source')
        
        plt.subplot(132)
        plt.imshow(image, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Lensed Image')
        
        plt.subplot(133)
        plt.imshow(kappa, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Convergence κ')
        
        plt.tight_layout()
        plt.show()
        
        return lens_model, kwargs_lens, image, unlensed_image
        
    except Exception as e:
        import traceback
        print(f"   ❌ ERROR: {e}")
        print(traceback.format_exc())
        return None, None, None, None


# Function to test scaled versions of the potential
def test_scaled_potential(x_grid, y_grid, potential, alpha_x, alpha_y, scale_factors=[10, 100, 1000, 10000]):
    """
    Test different scaling factors for the potential to find one that works
    """
    print("\n=== Testing Different Scaling Factors ===")
    
    for scale in scale_factors:
        print(f"\n--- Scale factor: {scale} ---")
        
        # Scale the potential and deflections
        scaled_potential = potential * scale
        scaled_alpha_x = alpha_x * scale
        scaled_alpha_y = alpha_y * scale
        
        # Run the debug function
        lens_model, kwargs_lens, image, unlensed_image = debug_interpol_with_auto_second_derivatives(
            x_grid, y_grid, scaled_potential, scaled_alpha_x, scaled_alpha_y
        )
        
        if lens_model is not None:
            print(f"\nScale factor {scale} completed - check the results above.")
            
            if image is not None and unlensed_image is not None:
                ratio = np.max(image) / np.max(unlensed_image)
                if abs(ratio - 1) > 0.05:  # More than 5% difference
                    print(f"✓ Scale factor {scale} shows significant lensing effect (ratio: {ratio:.3f})")
                    print(f"👉 SUGGESTION: Use this scale factor in your notebook")
                    return scale, lens_model, kwargs_lens
    
    print("\n❌ None of the scale factors produced significant lensing effects.")
    print("👉 Check your potential calculation or try even larger scale factors.")
    return None, None, None


# Example usage - this can be copied into the Jupyter notebook
"""
# Run the debug function on your data
lens_model, kwargs_lens, image, unlensed = debug_interpol_with_auto_second_derivatives(
    x_2d_arcsec,   # X grid in arcseconds
    y_2d_arcsec,   # Y grid in arcseconds
    lensing_potential,  # Lensing potential (dimensionless)
    alpha_x_custom,     # X deflection angle (arcsec)
    alpha_y_custom      # Y deflection angle (arcsec)
)

# If the lensing effect is too small, try scaling the potential
if lens_model is None or (image is not None and abs(np.max(image)/np.max(unlensed) - 1) < 0.05):
    print("\nTrying scaled versions of the potential...")
    scale, lens_model, kwargs_lens = test_scaled_potential(
        x_2d_arcsec, y_2d_arcsec, lensing_potential, alpha_x_custom, alpha_y_custom,
        scale_factors=[10, 100, 1000, 10000]
    )
"""
