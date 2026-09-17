#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced STL Generation with NACA Profile Integration
Generates complete 3D STL models with defined NACA profile
Not just the planform, but the entire wing geometry
"""

import numpy as np
from stl import mesh, Mode
import sys

def wing_to_stl_with_naca_profile(vlm_object, output_filename='wing_with_profile.stl', 
                                  naca_resolution=50, format='binary'):
    """
    Generates COMPLETE STL file with NACA profile from VLM object.

    This method overcomes the limitation that VLM only generates the planform.
    Here we regenerate the 3D geometry with the complete NACA profile.

    Parameters:
    -----------
    vlm_object : VLM
        VLM object with wing configuration
    output_filename : str
        Output STL file name
    naca_resolution : int
        Number of points to discretize the NACA profile
    format : str
        'binary' or 'ascii'

    Returns:
    --------
    mesh_object : stl.mesh.Mesh
        numpy-stl mesh object
    """

    from lib.naca import naca_airfoil

    plane = vlm_object.plane
    wing_sections = plane['wing_sections']

    all_faces = []
    all_vertices = []
    vertex_offset = 0

    # ========================================================================
    # STEP 1: Generate 3D geometry with complete NACA profiles
    # ========================================================================

    y_current = 0
    z_current = 0
    x_lead_current = 0

    print("\n=== Generating 3D geometry with NACA profiles ===")

    for sec_idx, section in enumerate(wing_sections):
        print(f"\nSection {sec_idx + 1}:")

        # Section parameters
        span = section['span_fraction']
        chord_root = section['chord_root']
        chord_tip = section['chord_tip']
        sweep = section['sweep']
        dihedral = section['dihedral']
        NACA_root = section['NACA_root']
        NACA_tip = section['NACA_tip']
        twist_root = section.get('twist_root', 0)
        twist_tip = section.get('twist_tip', 0)

        print(f"  Span: {span:.2f}m")
        print(f"  NACA root: {NACA_root}, NACA tip: {NACA_tip}")
        print(f"  Twist root: {np.degrees(twist_root):.2f}°, Twist tip: {np.degrees(twist_tip):.2f}°")

        # ====================================================================
        # Generate NACA profiles at root and tip
        # ====================================================================

        x_root, z_upper_root, z_lower_root, _, _, _ = naca_airfoil(
            NACA_root, chord_root, twist_root, naca_resolution)

        x_tip, z_upper_tip, z_lower_tip, _, _, _ = naca_airfoil(
            NACA_tip, chord_tip, twist_tip, naca_resolution)

        # Combine upper and lower surfaces in leading edge order
        # Lower surface backwards + upper surface forward
        x_root_full = np.concatenate([x_root[::-1], x_root[1:]])
        z_root_full = np.concatenate([z_lower_root[::-1], z_upper_root[1:]])

        x_tip_full = np.concatenate([x_tip[::-1], x_tip[1:]])
        z_tip_full = np.concatenate([z_lower_tip[::-1], z_upper_tip[1:]])

        # ====================================================================
        # Calculate 3D positions for this section
        # ====================================================================

        # Next position considering sweep and dihedral
        y_next = y_current + span * np.cos(dihedral)
        z_next = z_current + span * np.sin(dihedral)
        x_lead_next = x_lead_current + span * np.tan(sweep)

        # ====================================================================
        # Create vertices by interpolating between root and tip
        # ====================================================================

        n_profile = len(x_root_full)
        print(f"  Points per profile: {n_profile}")

        # Store vertices for both sections (root and tip)
        vertices_root_section = []
        vertices_tip_section = []

        for i in range(n_profile):
            # Vertex at ROOT
            vertex_root = [
                x_root_full[i] + x_lead_current,  # x: local profile coordinate + sweep offset
                y_current,                        # y: span
                z_root_full[i] + z_current        # z: profile height + dihedral offset
            ]
            vertices_root_section.append(vertex_root)

            # Vertex at TIP
            vertex_tip = [
                x_tip_full[i] + x_lead_next,      # x: interpolated
                y_next,                           # y: next span
                z_tip_full[i] + z_next            # z: interpolated
            ]
            vertices_tip_section.append(vertex_tip)

        all_vertices.extend(vertices_root_section)
        all_vertices.extend(vertices_tip_section)

        # ====================================================================
        # Create triangular faces connecting both sections
        # ====================================================================

        for i in range(n_profile - 1):
            # Indices of the four vertices of the panel
            # Root row
            v_root_0 = vertex_offset + i
            v_root_1 = vertex_offset + i + 1

            # Tip row
            v_tip_0 = vertex_offset + n_profile + i
            v_tip_1 = vertex_offset + n_profile + i + 1

            # Create two triangles for this quadrilateral
            # Triangle 1: v_root_0 → v_root_1 → v_tip_0
            all_faces.append([v_root_0, v_root_1, v_tip_0])

            # Triangle 2: v_root_1 → v_tip_1 → v_tip_0
            all_faces.append([v_root_1, v_tip_1, v_tip_0])

        # Update offset for next section
        vertex_offset = len(all_vertices)

        # Update positions for next section
        y_current = y_next
        z_current = z_next
        x_lead_current = x_lead_next

        print(f"  Vertices in this section: {n_profile * 2}")
        print(f"  Faces generated: {(n_profile - 1) * 2}")

    # ========================================================================
    # STEP 2: Process symmetric wings
    # ========================================================================

    vertices = np.array(all_vertices)
    faces = np.array(all_faces)

    if plane.get('symmetric', True):
        print("\n=== Generating symmetric wing ===")

        # Create left side by reflecting over y = 0
        vertices_left = vertices.copy()
        vertices_left[:, 1] *= -1  # Reflect in y (change sign of span)

        # Combine vertices from both sides
        n_right = len(vertices)
        vertices = np.vstack([vertices, vertices_left])

        # Create faces for left side (reverse order for correct normal)
        faces_left = []
        for face in faces:
            faces_left.append([
                face[0] + n_right,
                face[2] + n_right,  # Reverse order to point outward
                face[1] + n_right
            ])

        all_faces = np.vstack([faces, np.array(faces_left)])

        print(f"  Total vertices: {len(vertices)}")
        print(f"  Total faces: {len(all_faces)}")

    # ========================================================================
    # STEP 3: Create mesh and export to STL
    # ========================================================================

    print(f"\n=== Creating STL mesh ===")

    # Create mesh using numpy-stl
    wing_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))

    for i, face in enumerate(all_faces):
        for j in range(3):
            wing_mesh.vectors[i][j] = vertices[face[j]]

    # Save file
    if format.lower() == 'binary':
        wing_mesh.save(output_filename, mode=Mode.BINARY)
        print(f"✓ Format: BINARY (compact)")
    else:
        wing_mesh.save(output_filename, mode=Mode.ASCII)
        print(f"✓ Format: ASCII (readable)")

    print(f"✓ STL file generated: {output_filename}")
    print(f"  Total triangles: {len(all_faces)}")
    print(f"  Total vertices: {len(vertices)}")

    # Calculate approximate volume
    volume = 0
    for i in range(len(all_faces)):
        v0 = wing_mesh.vectors[i][0]
        v1 = wing_mesh.vectors[i][1]
        v2 = wing_mesh.vectors[i][2]
        volume += np.dot(v0, np.cross(v1, v2))
    volume = np.abs(volume) / 6.0
    print(f"  Approximate volume: {volume:.4f} m³")

    return wing_mesh


def wing_to_stl_with_naca_config(plane_config, output_filename='wing_with_profile.stl',
                                 naca_resolution=100, format='binary'):
    """
    Generates STL with NACA profile from configuration (without using existing VLM).

    Parameters:
    -----------
    plane_config : dict
        Dictionary with 'wing_sections'
    output_filename : str
        Output STL file name
    naca_resolution : int
        Number of points to discretize the NACA profile
    format : str
        'binary' or 'ascii'

    Returns:
    --------
    mesh_object : stl.mesh.Mesh
        numpy-stl mesh object
    """

    from naca import naca_airfoil

    wing_sections = plane_config['wing_sections']

    all_faces = []
    all_vertices = []
    vertex_offset = 0

    print("\n=== Generating STL with NACA profiles (without VLM) ===")

    y_current = 0
    z_current = 0
    x_lead_current = 0

    for sec_idx, section in enumerate(wing_sections):
        print(f"Section {sec_idx + 1}: {section['NACA_root']} → {section['NACA_tip']}")

        # Parameters
        span = section['span_fraction']
        chord_root = section['chord_root']
        chord_tip = section['chord_tip']
        sweep = section['sweep']
        dihedral = section['dihedral']
        NACA_root = section['NACA_root']
        NACA_tip = section['NACA_tip']
        twist_root = section.get('twist_root', 0)
        twist_tip = section.get('twist_tip', 0)

        # Generate NACA profiles
        x_root, z_upper_root, z_lower_root, _, _, _ = naca_airfoil(
            NACA_root, chord_root, twist_root, naca_resolution)
        x_tip, z_upper_tip, z_lower_tip, _, _, _ = naca_airfoil(
            NACA_tip, chord_tip, twist_tip, naca_resolution)

        # Combine surfaces
        x_root_full = np.concatenate([x_root[::-1], x_root[1:]])
        z_root_full = np.concatenate([z_lower_root[::-1], z_upper_root[1:]])

        x_tip_full = np.concatenate([x_tip[::-1], x_tip[1:]])
        z_tip_full = np.concatenate([z_lower_tip[::-1], z_upper_tip[1:]])

        # Calculate 3D positions
        y_next = y_current + span * np.cos(dihedral)
        z_next = z_current + span * np.sin(dihedral)
        x_lead_next = x_lead_current + span * np.tan(sweep)

        n_profile = len(x_root_full)

        # Create vertices
        for i in range(n_profile):
            vertex_root = [
                x_root_full[i] + x_lead_current,
                y_current,
                z_root_full[i] + z_current
            ]
            vertex_tip = [
                x_tip_full[i] + x_lead_next,
                y_next,
                z_tip_full[i] + z_next
            ]
            all_vertices.extend([vertex_root, vertex_tip])

        # Create faces
        for i in range(n_profile - 1):
            v_root_0 = vertex_offset + 2 * i
            v_root_1 = vertex_offset + 2 * i + 2
            v_tip_0 = vertex_offset + 2 * i + 1
            v_tip_1 = vertex_offset + 2 * i + 3

            all_faces.append([v_root_0, v_root_1, v_tip_0])
            all_faces.append([v_root_1, v_tip_1, v_tip_0])

        vertex_offset = len(all_vertices)

        y_current = y_next
        z_current = z_next
        x_lead_current = x_lead_next

    # Process symmetry
    vertices = np.array(all_vertices)
    faces = np.array(all_faces)

    if plane_config.get('symmetric', True):
        vertices_left = vertices.copy()
        vertices_left[:, 1] *= -1

        n_right = len(vertices)
        vertices = np.vstack([vertices, vertices_left])

        faces_left = []
        for face in faces:
            faces_left.append([
                face[0] + n_right,
                face[2] + n_right,
                face[1] + n_right
            ])

        all_faces = np.vstack([faces, np.array(faces_left)])

    # Create mesh and export
    wing_mesh = mesh.Mesh(np.zeros(len(all_faces), dtype=mesh.Mesh.dtype))

    for i, face in enumerate(all_faces):
        for j in range(3):
            wing_mesh.vectors[i][j] = vertices[face[j]]

    # Save
    if format.lower() == 'binary':
        wing_mesh.save(output_filename, mode=Mode.BINARY)
    else:
        wing_mesh.save(output_filename, mode=Mode.ASCII)

    print(f"\n✓ STL file generated: {output_filename}")
    print(f"  Triangles: {len(all_faces)}")
    print(f"  Vertices: {len(vertices)}")

    return wing_mesh
