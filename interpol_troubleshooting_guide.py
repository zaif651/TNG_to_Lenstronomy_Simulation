"""
IllustrisTNG + lenstronomy INTERPOL Troubleshooting Guide

Based on the conversation history, you have:
1. Calculated 3D gravitational potential from IllustrisTNG data
2. Projected to 2D lensing potential 
3. Converted coordinates from kpc to arcseconds
4. Set up INTERPOL with custom potential and deflection angles
5. But not seeing lensing images in your simulation

Here are the most likely issues and solutions:
"""

import numpy as np

# ============================================================================
# ISSUE 1: POTENTIAL MAGNITUDE TOO SMALL
# ============================================================================

"""
Problem: Your lensing potential values may be too small to produce visible lensing

Check:
- Print np.abs(your_lensing_potential).max()  
- Should be on order of 1e-6 to 1e-3 for typical strong lenses
- If much smaller (< 1e-8), lensing will be negligible

Solutions:
A) Verify your lensing potential calculation:
   ψ = (4πG/c²) * ∫ ρ(x',y',z') * |z'| dz'
   - Check if you included the (4πG/c²) factor
   - Ensure proper integration along line of sight
   - Convert to dimensionless form for lenstronomy

B) Test with scaled potential:
   f_test = your_lensing_potential * 100  # Scale factor
   # If this works, then original potential was too weak
"""

# Example scaling test:
def test_potential_scaling(f_original, scale_factors=[1, 10, 100, 1000]):
    """Test different scaling factors for the lensing potential"""
    print("Testing potential scaling:")
    for scale in scale_factors:
        f_scaled = f_original * scale
        max_potential = np.abs(f_scaled).max()
        print(f"  Scale {scale:4d}: max |ψ| = {max_potential:.2e}")
        
        # Rule of thumb: need |ψ| > 1e-6 for detectable lensing
        if max_potential > 1e-6:
            print(f"    ✓ Potentially strong enough for lensing")
        else:
            print(f"    ⚠️  May be too weak")

# ============================================================================
# ISSUE 2: DEFLECTION ANGLE CALCULATION ERROR
# ============================================================================

"""
Problem: Deflection angles (f_x, f_y) calculated incorrectly from potential

Common mistakes:
1. Wrong sign: α = -∇ψ (note the minus sign!)
2. Wrong pixel spacing in gradient calculation
3. Coordinate system issues

Correct calculation:
"""

def calculate_deflection_angles_correct(lensing_potential, pixel_scale_arcsec):
    """
    Calculate deflection angles from lensing potential
    
    Parameters:
    -----------
    lensing_potential : 2D array
        Dimensionless lensing potential ψ
    pixel_scale_arcsec : float  
        Pixel scale in arcseconds
    
    Returns:
    --------
    alpha_x, alpha_y : 2D arrays
        Deflection angles in arcseconds
    """
    import numpy as np
    
    # Calculate gradients (note: numpy gradient has spacing in denominator)
    grad_y, grad_x = np.gradient(lensing_potential, pixel_scale_arcsec)
    
    # Deflection angle = -gradient of potential (IMPORTANT: minus sign!)
    alpha_x = -grad_x  # f_x in INTERPOL
    alpha_y = -grad_y  # f_y in INTERPOL
    
    return alpha_x, alpha_y

# ============================================================================  
# ISSUE 3: COORDINATE SYSTEM PROBLEMS
# ============================================================================

"""
Problem: Coordinate grid setup doesn't match lenstronomy conventions

Requirements:
1. grid_interp_x, grid_interp_y should be meshgrids in arcseconds
2. Origin (0,0) should be at lens center
3. Coordinate order should match your potential array indexing

Check your coordinate setup:
"""

def verify_coordinate_setup(grid_x, grid_y, potential):
    """Verify coordinate grid setup"""
    print("Coordinate system check:")
    print(f"  Grid shapes: x={grid_x.shape}, y={grid_y.shape}, ψ={potential.shape}")
    print(f"  X range: [{grid_x.min():.2f}, {grid_x.max():.2f}] arcsec")  
    print(f"  Y range: [{grid_y.min():.2f}, {grid_y.max():.2f}] arcsec")
    
    # Check if origin is included
    x_includes_zero = (grid_x.min() <= 0) and (grid_x.max() >= 0)
    y_includes_zero = (grid_y.min() <= 0) and (grid_y.max() >= 0)
    print(f"  Includes origin: x={x_includes_zero}, y={y_includes_zero}")
    
    if not (x_includes_zero and y_includes_zero):
        print("  ⚠️  WARNING: Grid doesn't include origin (0,0)")
        print("     This may cause issues if sources are placed near center")

# ============================================================================
# ISSUE 4: SECOND DERIVATIVES (HESSIAN) ERRORS  
# ============================================================================

"""
Problem: Incorrect second derivative calculation for f_xx, f_yy, f_xy

These are needed for magnification and convergence calculations.
"""

def calculate_hessian_correct(lensing_potential, pixel_scale_arcsec):
    """Calculate second derivatives of lensing potential"""
    import numpy as np
    
    # First derivatives  
    grad1_y, grad1_x = np.gradient(lensing_potential, pixel_scale_arcsec)
    
    # Second derivatives
    grad2_yy, grad2_yx = np.gradient(grad1_y, pixel_scale_arcsec)  
    grad2_xy, grad2_xx = np.gradient(grad1_x, pixel_scale_arcsec)
    
    return grad2_xx, grad2_yy, grad2_xy

# ============================================================================
# ISSUE 5: SIMULATION SETUP PROBLEMS
# ============================================================================

"""
Problem: Source placement or imaging parameters incompatible with lens

Common issues:
1. Source placed outside lens caustics (no multiple images)
2. Image resolution too coarse to resolve lensing features  
3. Source too faint compared to noise
4. Field of view doesn't cover image positions

Solutions:
"""

def create_test_simulation_setup():
    """Create a robust test setup for INTERPOL lens"""
    
    # High resolution imaging
    kwargs_data = {
        'nx': 200, 'ny': 200,           # High resolution
        'pixel_scale': 0.05,            # Fine pixel scale
        'ra_at_xy0': 0.0,              # Centered on lens  
        'dec_at_xy0': 0.0
    }
    
    # Bright source near caustic
    kwargs_source = [{
        'amp': 10000,                   # Bright source
        'sigma': 0.2,                   # Compact  
        'center_x': 0.1,               # Slightly off-center
        'center_y': 0.05               # Should be lensed
    }]
    
    return kwargs_data, kwargs_source

# ============================================================================
# DIAGNOSTIC WORKFLOW
# ============================================================================

def run_full_diagnostic(grid_x, grid_y, f_, f_x, f_y, f_xx, f_yy, f_xy):
    """Complete diagnostic workflow"""
    
    print("=" * 60)
    print("INTERPOL DIAGNOSTIC WORKFLOW")  
    print("=" * 60)
    
    # 1. Basic checks
    print("\n1. BASIC PARAMETER CHECKS")
    print(f"   Potential range: [{f_.min():.2e}, {f_.max():.2e}]")
    print(f"   Max |deflection|: {np.sqrt(f_x**2 + f_y**2).max():.2e} arcsec")
    
    # 2. Coordinate verification
    print("\n2. COORDINATE SYSTEM")
    verify_coordinate_setup(grid_x, grid_y, f_)
    
    # 3. Physics checks
    print("\n3. LENSING PHYSICS")
    kappa = 0.5 * (f_xx + f_yy)
    print(f"   Convergence range: [{kappa.min():.2e}, {kappa.max():.2e}]")
    print(f"   Max |κ|: {np.abs(kappa).max():.2e}")
    
    # 4. Potential scaling test
    print("\n4. SCALING TEST")
    test_potential_scaling(f_)
    
    # 5. INTERPOL model test
    print("\n5. MODEL CREATION TEST")
    try:
        from lenstronomy.LensModel.lens_model import LensModel
        lens_model = LensModel(['INTERPOL'])
        kwargs_lens = [{'grid_interp_x': grid_x, 'grid_interp_y': grid_y,
                       'f_': f_, 'f_x': f_x, 'f_y': f_y,
                       'f_xx': f_xx, 'f_yy': f_yy, 'f_xy': f_xy}]
        
        # Test evaluation
        alpha_x, alpha_y = lens_model.alpha(0.0, 0.0, kwargs_lens)
        print(f"   ✓ Model created successfully")
        print(f"   ✓ Deflection at origin: ({alpha_x:.2e}, {alpha_y:.2e})")
        
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
    return True

# ============================================================================
# QUICK FIX TEMPLATES
# ============================================================================

print("""
QUICK FIX CHECKLIST:
□ 1. Scale test: Multiply potential by 10-1000 and test
□ 2. Sign check: Ensure deflection = -gradient(potential)  
□ 3. Units check: Potential dimensionless, deflections in arcsec
□ 4. Origin check: Grid coordinates include (0,0)
□ 5. Resolution test: Try finer pixel scale (0.01-0.05 arcsec)
□ 6. Bright source: Use amp > 1000 for testing
□ 7. Compare with SIE: Test same setup with analytical lens

If still not working, the issue is likely in steps 1-3 above.
""")
