"""
EJEMPLO INTEGRADO: Generación de STL con Perfil NACA Completo
Uso del script wing_to_stl_with_naca_ENHANCED.py
"""

import numpy as np


def example_1_from_vlm_complete():
    """
    Ejemplo 1: Generar STL desde objeto VLM COMPLETO
    Incluye análisis aerodinámico + geometría 3D con perfil NACA
    """

    from vlm import VLM
    from wing_to_stl_improved import wing_to_stl_with_naca_profile

    print("\n" + "="*70)
    print("EJEMPLO 1: STL Completo desde VLM (con análisis aerodinámico)")
    print("="*70)

    # Configuración del avión
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

    # Crear y ejecutar VLM
    print("\n1. Ejecutando análisis VLM...")
    vlm = VLM(plane, u=50, rho=1.225, alpha=np.radians(5), beta=0, n=20, m=10)
    vlm.calculate_geometry()
    vlm.calculate_discretization()
    vlm.calculate_wing_lift()

    print(f"   ✓ Wing lift: {vlm.CL:.4f}")
    print(f"   ✓ Wing area: {vlm.wing_area:.4f} m²")

    # Generar STL con perfil NACA
    print("\n2. Generando STL con perfil NACA completo...")
    wing_mesh = wing_to_stl_with_naca_profile(
        vlm,
        output_filename='01_ala_vlm_con_naca.stl',
        naca_resolution=80,
        format='binary'
    )

    print("\n✓ Archivo generado: 01_ala_vlm_con_naca.stl")

    return vlm, wing_mesh


def example_2_quick_generation():
    """
    Ejemplo 2: Generación RÁPIDA sin ejecutar VLM
    Solo geometría 3D con perfil NACA
    """

    from wing_to_stl_improved import wing_to_stl_with_naca_config

    print("\n" + "="*70)
    print("EJEMPLO 2: STL Rápido (sin VLM)")
    print("="*70)

    # Configuración simple
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

    print("\n1. Generando STL rápido...")
    wing_mesh = wing_to_stl_with_naca_config(
        plane_config,
        output_filename='02_ala_rapida_con_naca.stl',
        naca_resolution=100,
        format='binary'
    )

    print("\n✓ Archivo generado: 02_ala_rapida_con_naca.stl")

    return wing_mesh


def example_3_complex_wing():
    """
    Ejemplo 3: Ala compleja con múltiples secciones y variación de perfil
    """

    from wing_to_stl_improved import wing_to_stl_with_naca_config

    print("\n" + "="*70)
    print("EJEMPLO 3: Ala Compleja con Múltiples Secciones")
    print("="*70)

    plane_config = {
        'wing_sections': [
            # Sección raíz (fuselaje)
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
            # Sección media
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
            # Sección punta (winglet)
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

    print("\n1. Generando ala compleja...")
    wing_mesh = wing_to_stl_with_naca_config(
        plane_config,
        output_filename='03_ala_compleja_con_naca.stl',
        naca_resolution=120,
        format='binary'
    )

    print("\n✓ Archivo generado: 03_ala_compleja_con_naca.stl")

    return wing_mesh


def example_4_comparison():
    """
    Ejemplo 4: Comparación de resoluciones NACA
    Permite visualizar el impacto de la resolución
    """

    from wing_to_stl_improved import wing_to_stl_with_naca_config

    print("\n" + "="*70)
    print("EJEMPLO 4: Comparación de Resoluciones NACA")
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

    print("\n1. Generando STL con diferentes resoluciones...")

    for res in resolutions:
        filename = f'04_ala_resolucion_{res}.stl'
        print(f"\n   Generando con resolución {res}...")

        wing_mesh = wing_to_stl_with_naca_config(
            plane_config,
            output_filename=filename,
            naca_resolution=res,
            format='binary'
        )

    print("\n✓ Archivos generados:")
    for res in resolutions:
        print(f"   • 04_ala_resolucion_{res}.stl")

    print("\n💡 Recomendación:")
    print("   • Resolución 50-80:  Buena relación geometría/tamaño")
    print("   • Resolución 100+:   Muy suave para impresión 3D")
    print("   • Resolución 30-40:  Rápido para visualización preliminar")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("EJEMPLOS: Generación de STL con Perfil NACA Completo")
    print("="*70)

    try:
        # Ejecutar ejemplos
        print("\n[Ejecutando Ejemplo 1]")
        example_1_from_vlm_complete()

        print("\n[Ejecutando Ejemplo 2]")
        example_2_quick_generation()

        print("\n[Ejecutando Ejemplo 3]")
        example_3_complex_wing()

        print("\n[Ejecutando Ejemplo 4]")
        example_4_comparison()

        # Resumen final
        print("\n" + "="*70)
        print("✅ TODOS LOS EJEMPLOS COMPLETADOS CON ÉXITO")
        print("="*70)

        print("\n📁 Archivos generados:")
        print("   1. 01_ala_vlm_con_naca.stl        - STL desde VLM")
        print("   2. 02_ala_rapida_con_naca.stl     - STL rápido")
        print("   3. 03_ala_compleja_con_naca.stl   - Ala con 3 secciones")
        print("   4. 04_ala_resolucion_*.stl        - Comparación de resoluciones")

        print("\n🎯 Próximos pasos:")
        print("   1. Abre los archivos en FreeCAD: File → Open")
        print("   2. O visualiza en: https://www.viewstl.com/")
        print("   3. Compara las diferentes resoluciones")

        print("\n💾 Para guardar tus propias alas:")
        print("   • Modifica 'plane_config' con tus parámetros")
        print("   • Llama a wing_to_stl_with_naca_config()")
        print("   • ¡Listo para importar en CAD o slicers!")

        print("\n" + "="*70)

    except Exception as e:
        print(f"\n✗ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
