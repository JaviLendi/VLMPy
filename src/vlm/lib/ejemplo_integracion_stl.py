# Author: Javier Lendínez Castillo
# SPDX-License-Identifier: AGPL-3.0-only

"""Integrated example: generate STL geometry with a complete NACA profile."""

import numpy as np


def example_1_from_vlm_complete():
    """
    Example 1: Generate an STL from a complete VLM object.
    Includes aerodynamic analysis and 3D geometry with a NACA profile.
    """

    from vlm import VLM
    from wing_to_stl import wing_to_stl_with_naca_profile

    print("\n" + "="*70)
    print("EXAMPLE 1: Complete STL from VLM (with aerodynamic analysis)")
    print("="*70)

    # Aircraft configuration.
    plane = {
        'wing_sections': [
            {
                'span_fraction': 6.0,
                'chord_root': 1.2,
                'chord_tip': 0.8,
                'sweep': np.radians(5),
                'dihedral': np.radians(3),
                'NACA_root': '2412',
                'NACA_tip': '2410',
                'twist_root': 0,
                'twist_tip': np.radians(-2)
            }
        ]
    }

    # Create and run the VLM solver.
    print("\n1. Running VLM analysis...")
    vlm = VLM(plane, u=50, rho=1.225, alpha=np.radians(5), beta=0, n=20, m=10)
    vlm.calculate_geometry()
    vlm.calculate_discretization()
    vlm.calculate_wing_lift()

    print(f"   ✓ Wing lift: {vlm.CL:.4f}")
    print(f"   ✓ Wing area: {vlm.wing_area:.4f} m²")

    # Generate the STL with a NACA profile.
    print("\n2. Generating complete STL with a NACA profile...")
    wing_mesh = wing_to_stl_with_naca_profile(
        vlm,
        output_filename='01_vlm_wing_with_naca.stl',
        naca_resolution=80,
        format='binary'
    )

    print("\n✓ File generated: 01_vlm_wing_with_naca.stl")

    return vlm, wing_mesh


def example_2_quick_generation():
    """
    Example 2: Quick generation without running VLM.
    Generates only 3D geometry with a NACA profile.
    """

    from wing_to_stl import wing_to_stl_with_naca_config

    print("\n" + "="*70)
    print("EXAMPLE 2: Quick STL (without VLM)")
    print("="*70)

    # Simple configuration.
    plane_config = {
        'wing_sections': [
            {
                'span_fraction': 6.0,
                'chord_root': 1.2,
                'chord_tip': 0.8,
                'sweep': np.radians(5),
                'dihedral': np.radians(3),
                'NACA_root': '2412',
                'NACA_tip': '2410',
                'twist_root': 0,
                'twist_tip': np.radians(-2)
            }
        ],
        'symmetric': True
    }

    print("\n1. Generating a quick STL...")
    wing_mesh = wing_to_stl_with_naca_config(
        plane_config,
        output_filename='02_quick_wing_with_naca.stl',
        naca_resolution=100,
        format='binary'
    )

    print("\n✓ File generated: 02_quick_wing_with_naca.stl")

    return wing_mesh


def example_3_complex_wing():
    """
    Example 3: A complex wing with multiple sections and airfoil variations.
    """

    from wing_to_stl import wing_to_stl_with_naca_config

    print("\n" + "="*70)
    print("EXAMPLE 3: Complex Wing with Multiple Sections")
    print("="*70)

    plane_config = {
        'wing_sections': [
            # Root section (fuselage).
            {
                'span_fraction': 2.0,
                'chord_root': 1.5,
                'chord_tip': 1.2,
                'sweep': np.radians(10),
                'dihedral': np.radians(0),
                'NACA_root': '4415',
                'NACA_tip': '4412',
                'twist_root': 0,
                'twist_tip': np.radians(-1)
            },
            # Mid-span section.
            {
                'span_fraction': 2.5,
                'chord_root': 1.2,
                'chord_tip': 0.9,
                'sweep': np.radians(10),
                'dihedral': np.radians(0),
                'NACA_root': '4412',
                'NACA_tip': '2410',
                'twist_root': np.radians(-1),
                'twist_tip': np.radians(-2)
            },
            # Tip section (winglet).
            {
                'span_fraction': 1.5,
                'chord_root': 0.9,
                'chord_tip': 0.2,
                'sweep': np.radians(15),
                'dihedral': np.radians(0),
                'NACA_root': '2410',
                'NACA_tip': '2408',
                'twist_root': np.radians(-2),
                'twist_tip': np.radians(-3)
            }
        ],
        'symmetric': True
    }

    print("\n1. Generating a complex wing...")
    wing_mesh = wing_to_stl_with_naca_config(
        plane_config,
        output_filename='03_complex_wing_with_naca.stl',
        naca_resolution=120,
        format='binary'
    )

    print("\n✓ File generated: 03_complex_wing_with_naca.stl")

    return wing_mesh


def example_4_comparison():
    """
    Example 4: Compare NACA resolutions and visualize their effects.
    """

    from wing_to_stl import wing_to_stl_with_naca_config

    print("\n" + "="*70)
    print("EXAMPLE 4: NACA Resolution Comparison")
    print("="*70)

    plane_config = {
        'wing_sections': [
            {
                'span_fraction': 4.0,
                'chord_root': 1.0,
                'chord_tip': 0.7,
                'sweep': np.radians(8),
                'dihedral': np.radians(3),
                'NACA_root': '2412',
                'NACA_tip': '2410',
                'twist_root': 0,
                'twist_tip': np.radians(-2)
            }
        ],
        'symmetric': True
    }

    resolutions = [30, 50, 80, 150]

    print("\n1. Generating STL files at different resolutions...")

    for res in resolutions:
        filename = f'04_wing_resolution_{res}.stl'
        print(f"\n   Generating at resolution {res}...")

        wing_mesh = wing_to_stl_with_naca_config(
            plane_config,
            output_filename=filename,
            naca_resolution=res,
            format='binary'
        )

    print("\n✓ Files generated:")
    for res in resolutions:
        print(f"   • 04_wing_resolution_{res}.stl")

    print("\n💡 Recommendations:")
    print("   • Resolution 50-80:  Good geometry-to-file-size tradeoff")
    print("   • Resolution 100+:   Smooth enough for 3D printing")
    print("   • Resolution 30-40:  Fast for preliminary visualization")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("EXAMPLES: STL Generation with a Complete NACA Profile")
    print("="*70)

    try:
        # Run the examples.
        print("\n[Running Example 1]")
        example_1_from_vlm_complete()

        print("\n[Running Example 2]")
        example_2_quick_generation()

        print("\n[Running Example 3]")
        example_3_complex_wing()

        print("\n[Running Example 4]")
        example_4_comparison()

        # Final summary.
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)

        print("\n📁 Files generated:")
        print("   1. 01_vlm_wing_with_naca.stl       - STL from VLM")
        print("   2. 02_quick_wing_with_naca.stl     - Quick STL")
        print("   3. 03_complex_wing_with_naca.stl   - Three-section wing")
        print("   4. 04_wing_resolution_*.stl        - Resolution comparison")

        print("\n🎯 Next steps:")
        print("   1. Open the files in FreeCAD: File → Open")
        print("   2. Or view them at: https://www.viewstl.com/")
        print("   3. Compare the different resolutions")

        print("\n💾 To save your own wings:")
        print("   • Update 'plane_config' with your parameters")
        print("   • Call wing_to_stl_with_naca_config()")
        print("   • Then import the files into CAD or a slicer")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
