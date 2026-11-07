#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Javier Lendínez
# SPDX-License-Identifier: AGPL-3.0-only

import sys
sys.path.append('../')  # Add parent directory to path for module imports

import time
import numpy as np
import plotly.graph_objects as go
from lib.naca import naca_airfoil_dzdx
from stl import mesh, Mode


def _calculate_section_geometry(sections, alpha, is_symmetric, x_translate, z_translate=0):
    """Module-level helper mirroring the nested calculate_section_geometry implementation.
    Kept as a helper to allow reuse and simpler testing/refactoring.
    """
    # Reuse the nested implementation by importing and executing it dynamically
    # to avoid duplicating logic. This helper exists as a stable API point for
    # refactors; callers may replace the nested version as needed.
    # For now, call the nested implementation by constructing an equivalent
    # function body using the original code path. (Minimal wrapper to preserve behavior.)
    # NOTE: The nested function currently lives inside calculate_geometry; to keep
    # behavior identical, we simply reimplement the same algorithm here.
    x_leading_edge = []
    y_leading_edge = []
    z_leading_edge = []
    x_trailing_edge = []
    y_trailing_edge = []
    z_trailing_edge = []
    x_quarter_chord = []
    y_quarter_chord = []
    z_quarter_chord = []
    twist_list = []

    y_current = 0
    z_current = 0
    x_lead_current = 0
    x_trail_current = sections[0]['chord_root']
    x_quarter_current = sections[0]['chord_tip'] / 4

    for i, sec in enumerate(sections):
        _validate_section(sec, i)
        span = sec['span_fraction']
        chord_tip = sec['chord_tip']
        sweep = sec['sweep']
        dihedral = sec['dihedral']

        span2 = span / 2 if is_symmetric else span

        y_next = y_current + (span2) * np.cos(dihedral)
        z_next = z_current + (span2) * np.sin(dihedral)

        if is_symmetric:
            x_lead_next = x_lead_current + (np.tan(sweep) * (span2)) * np.cos(dihedral)
        else:
            x_lead_next = x_lead_current + (np.tan(sweep) * (span2))

        x_trail_next = x_lead_next + chord_tip
        x_quarter_next = (x_trail_next - x_lead_next) / 4 + x_lead_next

        x_leading_edge.append(np.array([x_lead_current, x_lead_next]))
        y_leading_edge.append(np.array([y_current, y_next]))
        z_leading_edge.append(np.array([z_current, z_next]))

        x_trailing_edge.append(np.array([x_trail_current, x_trail_next]))
        y_trailing_edge.append(np.array([y_current, y_next]))
        z_trailing_edge.append(np.array([z_current, z_next]))

        x_quarter_chord.append(np.array([x_quarter_current, x_quarter_next]))
        y_quarter_chord.append(np.array([y_current, y_next]))
        z_quarter_chord.append(np.array([z_current, z_next]))
        twist_list.append(np.array([sec.get('twist', 0.0), sec.get('twist', 0.0)]))

        y_current = y_next
        z_current = z_next
        x_lead_current = x_lead_next
        x_trail_current = x_trail_next
        x_quarter_current = x_quarter_next

    x_leading_edge = np.concatenate(x_leading_edge)
    y_leading_edge = np.concatenate(y_leading_edge)
    z_leading_edge = np.concatenate(z_leading_edge)
    x_trailing_edge = np.concatenate(x_trailing_edge)
    y_trailing_edge = np.concatenate(y_trailing_edge)
    z_trailing_edge = np.concatenate(z_trailing_edge)
    x_quarter_chord = np.concatenate(x_quarter_chord)
    y_quarter_chord = np.concatenate(y_quarter_chord)
    z_quarter_chord = np.concatenate(z_quarter_chord)

    # Build twist array where twist is accumulated from the root: each section's
    # provided `twist` is interpreted as the incremental twist to be added at
    # the tip of that section. This produces a monotonic (cumulative) twist
    # distribution along the span endpoints.
    if len(twist_list) > 0:
        # twist_list currently holds per-segment [root_val, tip_val] pairs (section-local)
        # we will convert these into cumulative twist values.
        raw = np.concatenate(twist_list)
        # raw contains per-segment local twist increments duplicated; we need to
        # accumulate per section. We'll reconstruct by walking sections.
        cumulative = []
        cum = 0.0
        # twist_list was appended per section as [sec_twist, sec_twist]
        for sec_twist_pair in twist_list:
            sec_twist = float(sec_twist_pair[0]) if np.isfinite(sec_twist_pair[0]) else 0.0
            # root value for this section
            cumulative.append(cum)
            # tip value after applying this section's incremental twist
            cum = cum + sec_twist
            cumulative.append(cum)
        twist_array = np.array(cumulative)
        if twist_array.shape != x_quarter_chord.shape:
            # fallback: zero twist if shapes mismatch
            twist_array = np.zeros_like(x_quarter_chord)
    else:
        twist_array = np.zeros_like(x_quarter_chord)

    # Do not apply twist rotation here; return raw leading/trailing/quarter
    # positions and the cumulative `twist_array`. Rotation is applied later
    # in `interpolate_wing_points` per-sample to avoid double application.

    if alpha != 0:
        rotation_matrix = np.array([
            [np.cos(alpha), 0, np.sin(alpha)],
            [0, 1, 0],
            [-np.sin(alpha), 0, np.cos(alpha)]
        ])
        for coords in [
            (x_leading_edge, y_leading_edge, z_leading_edge),
            (x_trailing_edge, y_trailing_edge, z_trailing_edge),
            (x_quarter_chord, y_quarter_chord, z_quarter_chord)
        ]:
            rotated = rotation_matrix @ np.vstack(coords)
            coords[0][:] = rotated[0]
            coords[1][:] = rotated[1]
            coords[2][:] = rotated[2]

    x_leading_edge += x_translate
    x_trailing_edge += x_translate
    x_quarter_chord += x_translate

    z_leading_edge += z_translate
    z_trailing_edge += z_translate
    z_quarter_chord += z_translate

    if is_symmetric:
        x_leading_edge = np.concatenate((x_leading_edge[::-1], x_leading_edge))
        y_leading_edge = np.concatenate((-y_leading_edge[::-1], y_leading_edge))
        z_leading_edge = np.concatenate((z_leading_edge[::-1], z_leading_edge))

        x_trailing_edge = np.concatenate((x_trailing_edge[::-1], x_trailing_edge))
        y_trailing_edge = np.concatenate((-y_trailing_edge[::-1], y_trailing_edge))
        z_trailing_edge = np.concatenate((z_trailing_edge[::-1], z_trailing_edge))

        x_quarter_chord = np.concatenate((x_quarter_chord[::-1], x_quarter_chord))
        y_quarter_chord = np.concatenate((-y_quarter_chord[::-1], y_quarter_chord))
        z_quarter_chord = np.concatenate((z_quarter_chord[::-1], z_quarter_chord))
        # Mirror twist_array for symmetric surfaces so lengths match the quarter-chord arrays
        twist_array = np.concatenate((twist_array[::-1], twist_array))

    return {
        'leading_edge': (x_leading_edge, y_leading_edge, z_leading_edge),
        'trailing_edge': (x_trailing_edge, y_trailing_edge, z_trailing_edge),
        'quarter_chord': (x_quarter_chord, y_quarter_chord, z_quarter_chord),
        'twist': twist_array
    }


def _validate_section(sec, idx=0):
    """Validate a single section dict for required keys and numeric values.

    Raises ValueError with clear message on failure.
    """
    if not isinstance(sec, dict):
        raise ValueError(f"Section at index {idx} must be a dict, got {type(sec)}")

    required_keys = ['span_fraction', 'chord_root', 'chord_tip', 'sweep', 'dihedral']
    for k in required_keys:
        if k not in sec:
            raise ValueError(f"Section {idx} missing required key '{k}'")
        v = sec[k]
        if not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"Section {idx} key '{k}' must be a finite number, got {v}")

    # Additional simple physical checks
    if sec['span_fraction'] <= 0:
        raise ValueError(f"Section {idx} 'span_fraction' must be > 0")
    if sec['chord_root'] <= 0 or sec['chord_tip'] <= 0:
        raise ValueError(f"Section {idx} chord values must be > 0")

    # Optional numeric keys
    if 'twist' in sec and (not isinstance(sec['twist'], (int, float)) or not np.isfinite(sec['twist'])):
        raise ValueError(f"Section {idx} optional key 'twist' must be a finite number if provided")

def calculate_geometry(self):
    """
    Calculates the coordinates of the leading edge, trailing edge, and quarter chord line for a surface.
    Inputs:
    - plane: Dictionary containing wing sections and stabilizers
    - alpha: Angle of attack or incidence angle (rad)
    - apply_alpha_rotation: Boolean, whether to apply rotation around the y-axis
    - is_symmetric: Boolean, whether the surface is symmetric about the x-z plane
    - x_translate: Translation in the x-direction (m), default is 0
    Outputs:
    - surface_geometry: Dictionary with coordinates of leading edge, trailing edge, and quarter chord
    """
    plane = self.plane
    if plane is None or not isinstance(plane, dict):
        raise ValueError("self.plane must be a non-empty dict with wing/stabilizer definitions")
    def calculate_section_geometry(sections, alpha, is_symmetric, x_translate, z_translate=0):
        print(f"alpha: {alpha}")
        # Initialize coordinate arrays for one side
        x_leading_edge = []
        y_leading_edge = []
        z_leading_edge = []
        x_trailing_edge = []
        y_trailing_edge = []
        z_trailing_edge = []
        x_quarter_chord = []
        y_quarter_chord = []
        z_quarter_chord = []
        # Per-segment geometric twist (radians). Each segment contributes two endpoints.
        twist_list = []

        # Starting positions at the root
        y_current = 0
        z_current = 0
        x_lead_current = 0
        x_trail_current = sections[0]['chord_root']
        x_quarter_current = sections[0]['chord_tip'] / 4

        # Calculate coordinates for each section (one side)
        for sec in sections:
            span = sec['span_fraction']
            chord_tip = sec['chord_tip']
            sweep = sec['sweep']
            dihedral = sec['dihedral']
            chord_root = sec['chord_root']

            if is_symmetric:
                span2 = span / 2
            else:
                span2 = span
    
            # Next position along span, accounting for dihedral
            y_next = y_current + (span2) * np.cos(dihedral)
            z_next = z_current + (span2) * np.sin(dihedral)

            # X-coordinates based on sweep and chord
            if is_symmetric:
                x_lead_next = x_lead_current + (np.tan(sweep) * (span2)) * np.cos(dihedral)
            else:
                x_lead_next = x_lead_current + (np.tan(sweep) * (span2)) 

            x_trail_next = x_lead_next + chord_tip 
            x_quarter_next = (x_trail_next - x_lead_next) / 4 + x_lead_next

            # Store segment coordinates
            x_leading_edge.append(np.array([x_lead_current, x_lead_next]))
            y_leading_edge.append(np.array([y_current, y_next]))
            z_leading_edge.append(np.array([z_current, z_next]))

            x_trailing_edge.append(np.array([x_trail_current, x_trail_next]))
            y_trailing_edge.append(np.array([y_current, y_next]))
            z_trailing_edge.append(np.array([z_current, z_next]))

            x_quarter_chord.append(np.array([x_quarter_current, x_quarter_next]))
            y_quarter_chord.append(np.array([y_current, y_next]))
            z_quarter_chord.append(np.array([z_current, z_next]))
            # store twist increment for this section (allow 'twist' key in section, default 0)
            # Note: here `sec.get('twist')` is treated as the incremental twist to
            # apply across the section (radians). We'll accumulate these below.
            twist_list.append(np.array([sec.get('twist', 0.0), sec.get('twist', 0.0)]))

            # Update current positions
            y_current = y_next
            z_current = z_next
            x_lead_current = x_lead_next
            x_trail_current = x_trail_next
            x_quarter_current = x_quarter_next

        # Concatenate coordinates for one side
        x_leading_edge = np.concatenate(x_leading_edge)
        y_leading_edge = np.concatenate(y_leading_edge)
        z_leading_edge = np.concatenate(z_leading_edge)
        x_trailing_edge = np.concatenate(x_trailing_edge)
        y_trailing_edge = np.concatenate(y_trailing_edge)
        z_trailing_edge = np.concatenate(z_trailing_edge)
        x_quarter_chord = np.concatenate(x_quarter_chord)
        y_quarter_chord = np.concatenate(y_quarter_chord)
        z_quarter_chord = np.concatenate(z_quarter_chord)

        # Build cumulative twist distribution (per-endpoint) and rotate
        if len(twist_list) > 0:
            # twist_list contains per-section duplicated incremental twist values.
            cumulative = []
            cum = 0.0
            for sec_twist_pair in twist_list:
                sec_twist = float(sec_twist_pair[0]) if np.isfinite(sec_twist_pair[0]) else 0.0
                cumulative.append(cum)
                cum = cum + sec_twist
                cumulative.append(cum)
            twist_array = np.array(cumulative)
            if twist_array.shape != x_quarter_chord.shape:
                twist_array = np.zeros_like(x_quarter_chord)
        else:
            twist_array = np.zeros_like(x_quarter_chord)

        # Do not apply twist rotation here; return raw leading/trailing/quarter
        # positions and the cumulative `twist_array`. Rotation is applied later
        # in `interpolate_wing_points` per-sample to avoid double application.

        # Apply alpha rotation around y-axis if specified
        if alpha != 0:
            rotation_matrix = np.array([
                [np.cos(alpha), 0, np.sin(alpha)],
                [0, 1, 0],
                [-np.sin(alpha), 0, np.cos(alpha)]
            ])
            for coords in [
                (x_leading_edge, y_leading_edge, z_leading_edge),
                (x_trailing_edge, y_trailing_edge, z_trailing_edge),
                (x_quarter_chord, y_quarter_chord, z_quarter_chord)
            ]:
                rotated = rotation_matrix @ np.vstack(coords)
                coords[0][:] = rotated[0]
                coords[1][:] = rotated[1]
                coords[2][:] = rotated[2]

        # Apply x-translation
        x_leading_edge += x_translate
        x_trailing_edge += x_translate
        x_quarter_chord += x_translate

        # Apply y-translation
        z_leading_edge += z_translate
        z_trailing_edge += z_translate
        z_quarter_chord += z_translate

        # If symmetric, mirror across the x-z plane (y=0)
        if is_symmetric:
            x_leading_edge = np.concatenate((x_leading_edge[::-1], x_leading_edge))
            y_leading_edge = np.concatenate((-y_leading_edge[::-1], y_leading_edge))
            z_leading_edge = np.concatenate((z_leading_edge[::-1], z_leading_edge))

            x_trailing_edge = np.concatenate((x_trailing_edge[::-1], x_trailing_edge))
            y_trailing_edge = np.concatenate((-y_trailing_edge[::-1], y_trailing_edge))
            z_trailing_edge = np.concatenate((z_trailing_edge[::-1], z_trailing_edge))

            x_quarter_chord = np.concatenate((x_quarter_chord[::-1], x_quarter_chord))
            y_quarter_chord = np.concatenate((-y_quarter_chord[::-1], y_quarter_chord))
            z_quarter_chord = np.concatenate((z_quarter_chord[::-1], z_quarter_chord))
            # Mirror twist_array for symmetric surfaces so the return lengths match
            twist_array = np.concatenate((twist_array[::-1], twist_array))

        # Build twist_array for return (cumulative per-endpoint) to allow downstream
        # functions to apply a per-panel twist when interpolating finer-resolution
        # meshes. The twist_array was already calculated above.
        return {
            'leading_edge': (x_leading_edge, y_leading_edge, z_leading_edge),
            'trailing_edge': (x_trailing_edge, y_trailing_edge, z_trailing_edge),
            'quarter_chord': (x_quarter_chord, y_quarter_chord, z_quarter_chord),
            'twist': twist_array
        }

    wing_geometry = 0
    hs_geometry = 0
    vs_geometry = 0
    if 'wing_sections' in plane:
        if not isinstance(plane['wing_sections'], list) or len(plane['wing_sections']) == 0:
            raise ValueError("plane['wing_sections'] must be a non-empty list of section dicts")
        # Calculate geometry for wing sections
        wing_geometry = _calculate_section_geometry(plane['wing_sections'], self.alpha, is_symmetric=True, x_translate=0, z_translate=0)
    if 'horizontal_stabilizer' in plane:
        # Calculate geometry for horizontal stabilizer
        hs = plane['horizontal_stabilizer']
        if not isinstance(hs, dict):
            raise ValueError("plane['horizontal_stabilizer'] must be a dict describing the stabilizer")
        hs_geometry = _calculate_section_geometry([hs], hs.get('alpha', 0.0), True, hs.get('x_translate', 0.0), hs.get('z_translate', 0.0))
    if 'vertical_stabilizer' in plane:
        # Calculate geometry for vertical stabilizer
        vs = plane['vertical_stabilizer']
        if not isinstance(vs, dict):
            raise ValueError("plane['vertical_stabilizer'] must be a dict describing the stabilizer")
        vs_geometry = _calculate_section_geometry([vs], vs.get('alpha', 0.0), False, vs.get('x_translate', 0.0), vs.get('z_translate', 0.0))

    return wing_geometry, hs_geometry, vs_geometry

def interpolate_wing_points(self, type='wing'):
    """
    Interpolates points between the leading and trailing edges, and along the chord line.
    
    Parameters:
    - wing_geometry: Dict with 'leading_edge', 'trailing_edge', 'quarter_chord' as (x, y, z) arrays.
    - n: Number of points along the span.
    - m: Number of points along the chord.
    - type: 'wing' or other (e.g., 'vertical_stabilizer').
    
    Returns:
    - wing_vertical_lines: Array of shape (n, 4) with span coordinates and leading/trailing edge x.
    - wing_horizontal_lines: Array of shape (m, n) with x-coordinates along the chord.
    - wing_z_coordinates: Array of shape (m, n) with z (wing) or y (vertical stabilizer) coordinates.
    """
    if type == 'wing':
        n = self.n + 1
        m = self.m + 1
        geometry = self.wing_geometry
    elif type == 'hs':
        n = self.n_hs + 1
        m = self.m_hs + 1
        geometry = self.hs_geometry
    elif type == 'vs':
        n = self.n_vs + 1
        m = self.m_vs + 1
        geometry = self.vs_geometry

    leading_edge, trailing_edge, quarter_chord = (geometry[key] for key in ['leading_edge', 'trailing_edge', 'quarter_chord'])
    twist_array = geometry.get('twist', None)
    chord_fractions = np.linspace(0, 1, m)  # Parameter from leading (0) to trailing (1) edge
    
    if type == 'wing' or type == 'hs':
        # Span along y-axis
        span_leading, span_trailing = leading_edge[1], trailing_edge[1]
        
        y_leading   = np.linspace(span_leading[0], span_leading[-1], n) 
        y_trailing  = np.linspace(span_trailing[0], span_trailing[-1], n) 

        # Cosine interpolation for smoother transition
        #y_leading   = np.linspace(span_leading[0], span_leading[-1], n) * np.cos(np.linspace(0, np.pi, n))
        #y_trailing  = np.linspace(span_trailing[0], span_trailing[-1], n) * np.cos(np.linspace(0, np.pi, n))

        x_leading   = np.interp(y_leading, span_leading, leading_edge[0])
        x_trailing  = np.interp(y_trailing, span_trailing, trailing_edge[0])
        z_leading   = np.interp(y_leading, span_leading, leading_edge[2])
        z_trailing  = np.interp(y_trailing, span_trailing, trailing_edge[2])

        # If a twist distribution is provided (cumulative per-endpoint), compute
        # twist per interpolated span sample and rotate chord points about the
        # local quarter-chord. This allows per-panel twist that depends on the
        # number of span panels `n` used for discretization.
        if twist_array is not None:
            x_q = quarter_chord[0]
            y_q = quarter_chord[1]
            z_q = quarter_chord[2]
            # Interpolate quarter-chord position and twist to the sample y positions
            qx_at_y = np.interp(y_leading, y_q, x_q)
            qz_at_y = np.interp(y_leading, y_q, z_q)
            twist_at_y = np.interp(y_leading, y_q, twist_array)
        else:
            qx_at_y = np.zeros_like(y_leading)
            qz_at_y = np.zeros_like(y_leading)
            twist_at_y = np.zeros_like(y_leading)

        # Interpolate along chord for each span position (as 2D arrays m x n)
        wing_horizontal_lines = []
        wing_z_coordinates = []
        for s in chord_fractions:
            x_line = x_leading * (1 - s) + x_trailing * s
            z_line = z_leading * (1 - s) + z_trailing * s
            # Apply twist rotation per span sample
            for j in range(x_line.shape[0]):
                t = twist_at_y[j]
                if t != 0.0:
                    xr = x_line[j] - qx_at_y[j]
                    zr = z_line[j] - qz_at_y[j]
                    ct = np.cos(t); st = np.sin(t)
                    x_line[j] = ct * xr + st * zr + qx_at_y[j]
                    z_line[j] = -st * xr + ct * zr + qz_at_y[j]
            wing_horizontal_lines.append(x_line)
            wing_z_coordinates.append(z_line)

        wing_horizontal_lines = np.array(wing_horizontal_lines)
        wing_z_coordinates = np.array(wing_z_coordinates)
        wing_vertical_lines     = np.column_stack((y_leading, y_trailing, x_leading, x_trailing))
    
    elif type =='vs':  # Vertical stabilizer
        # Span along z-axis
        span_leading, span_trailing = leading_edge[2], trailing_edge[2]
        
        z_leading   = np.linspace(span_leading[0], span_leading[-1], n) 
        z_trailing  = np.linspace(span_trailing[0], span_trailing[-1], n) 
        
        # Interpolate leading and trailing edge coordinates at span positions
        x_leading   = np.interp(z_leading, span_leading, leading_edge[0])
        x_trailing  = np.interp(z_trailing, span_trailing, trailing_edge[0])
        y_leading   = np.interp(z_leading, span_leading, leading_edge[1])
        y_trailing  = np.interp(z_trailing, span_trailing, trailing_edge[1])
        
        # Interpolate along chord for each span position
        wing_horizontal_lines   = [x_leading * (1 - s) + x_trailing * s for s in chord_fractions]
        wing_z_coordinates      = [z_leading * (1 - s) + z_trailing * s for s in chord_fractions]
        wing_vertical_lines     = np.column_stack((y_leading, y_trailing, x_leading, x_trailing))
    
    return wing_vertical_lines, np.array(wing_horizontal_lines), np.array(wing_z_coordinates)

def generate_plane_panels(self):
    """
    Generates the panels of the entire plane (wing, horizontal stabilizer, and vertical stabilizer) using the coordinates of the points between the leading and trailing edges, and along the chord line in 3D.
    Each panel is defined by its four corner points:
    [x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]. 
    Also calculates the quarter chord lines and the midpoints at 3/4 of the chord.
    Inputs:
    - discretization: Dictionary containing the vertical, horizontal, and z points for wing, horizontal stabilizer, and vertical stabilizer
    Outputs:
    - panel_data: List with the information of each panel
    - panel_lengths: List with the lengths of each panel
    """
    discretization = self.discretization
    plane = self.plane
    def generate_panels_for_surface(surface_discretization, sections):
        panels = []
        quarter_chord_lines = []
        three_quarter_chord_midpoints = []
        panel_data = []
        normal_vectors = []

        vertical_lines = surface_discretization['vertical_points']
        horizontal_lines = surface_discretization['horizontal_points']
        z_points = surface_discretization['z_points']

        num_chordwise = len(horizontal_lines)
        num_spanwise = len(vertical_lines)
        points = np.zeros((num_chordwise, num_spanwise, 3))

        # Step 1: Populate the point grid
        for i in range(num_spanwise):
            y_i = vertical_lines[i, 0]  # Assuming vertical_lines[i, 0] = vertical_lines[i, 1]
            for j in range(num_chordwise):
                points[j, i] = [y_i, horizontal_lines[j, i], z_points[j, i]]

        # Step 2: Apply flap deflection
        span = vertical_lines[-1, 1] - vertical_lines[0, 0]
        half_span = span / 2

        for sec in sections:
            if 'flap_start' in sec:
                flap_start = sec['flap_start'] * half_span
                flap_end = sec['flap_end'] * half_span
                flap_hinge_chord = sec['flap_hinge_chord']
                deflection_angle = sec['deflection_angle']
                deflection_type = sec['deflection_type']

                for i in range(num_spanwise):
                    y = points[0, i][0]
                    if deflection_type == 'symmetrical':
                        if flap_start <= abs(y) <= flap_end:
                            chord_length = horizontal_lines[-1, i] - horizontal_lines[0, i]
                            x_hinge = horizontal_lines[0, i] + flap_hinge_chord * chord_length
                            for j in range(num_chordwise):
                                x_j = points[j, i][1]
                                if x_j > x_hinge:
                                    distance_from_hinge = x_j - x_hinge
                                    points[j, i][2] -= deflection_angle * distance_from_hinge
                    elif deflection_type == 'antisymmetrical':
                        if flap_start <= y <= flap_end:  # Right wing (y > 0)
                            chord_length = horizontal_lines[-1, i] - horizontal_lines[0, i]
                            x_hinge = horizontal_lines[0, i] + flap_hinge_chord * chord_length
                            for j in range(num_chordwise):
                                x_j = points[j, i][1]
                                if x_j > x_hinge:
                                    distance_from_hinge = x_j - x_hinge
                                    points[j, i][2] -= deflection_angle * distance_from_hinge
                        elif -flap_end <= y <= -flap_start:  # Left wing (y < 0)
                            chord_length = horizontal_lines[-1, i] - horizontal_lines[0, i]
                            x_hinge = horizontal_lines[0, i] + flap_hinge_chord * chord_length
                            for j in range(num_chordwise):
                                x_j = points[j, i][1]
                                if x_j > x_hinge:
                                    distance_from_hinge = x_j - x_hinge
                                    points[j, i][2] -= -deflection_angle * distance_from_hinge  # Opposite deflection

        # Step 3: Generate panels
        for i in range(num_spanwise - 1):
            for j in range(num_chordwise - 1):
                panel = [
                    points[j, i].copy(),
                    points[j + 1, i].copy(),
                    points[j + 1, i + 1].copy(),
                    points[j, i + 1].copy()
                ]
                panels.append(panel)

                # Calculate the quarter chord points using interpolation
                t_quarter = 0.25
                # First edge: points[j, i] to points[j + 1, i]
                P0 = points[j, i]
                P1 = points[j + 1, i]
                q_point1 = P0 + t_quarter * (P1 - P0)
                # Second edge: points[j, i + 1] to points[j + 1, i + 1]
                Q0 = points[j, i + 1]
                Q1 = points[j + 1, i + 1]
                q_point2 = Q0 + t_quarter * (Q1 - Q0)
                quarter_chord_lines.append([q_point1.tolist(), q_point2.tolist()])

                # Calculate the three-quarter chord points and midpoint
                t_three_quarter = 0.75
                # First edge
                three_q_point1 = P0 + t_three_quarter * (P1 - P0)
                # Second edge
                three_q_point2 = Q0 + t_three_quarter * (Q1 - Q0)
                # Midpoint is the average of the two points
                three_quarter_chord_midpoint = (three_q_point1 + three_q_point2) / 2
                three_quarter_chord_midpoints.append(three_quarter_chord_midpoint.tolist())

                # Calculate the normal vector of the panel at the midpoints at 3/4 of the chord
                panel = panels[-1]
                if panel[0][0] > 0:
                    v1 = np.array(panel[3]) - np.array(panel[1])
                    v2 = np.array(panel[2]) - np.array(panel[0])
                    normal_vec = (np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2)))
                else:
                    v1 = np.array(panel[1]) - np.array(panel[3])
                    v2 = np.array(panel[0]) - np.array(panel[2])
                    normal_vec = (np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2)))

                normal_vec = np.array([normal_vec[0], normal_vec[1], normal_vec[2]])
                normal_vectors.append([normal_vec, three_quarter_chord_midpoint])

        for idx, panel in enumerate(panels):
            panel_info = (idx + 1, panel, quarter_chord_lines[idx], three_quarter_chord_midpoints[idx], normal_vectors[idx])
            panel_data.append(panel_info)

        return panel_data

    if 'wing' in discretization:
        wing_panel_data = generate_panels_for_surface(discretization['wing'], plane['wing_sections'])
        panel_data = wing_panel_data
    if 'horizontal_stabilizer' in discretization:
        hs_panel_data = generate_panels_for_surface(discretization['horizontal_stabilizer'], [plane['horizontal_stabilizer']])
        panel_data.extend(hs_panel_data)
    if 'vertical_stabilizer' in discretization:
        vs_panel_data = generate_panels_for_surface(discretization['vertical_stabilizer'], [plane['vertical_stabilizer']])
        panel_data.extend(vs_panel_data)

    # Calculate the area of the panels 
    panel_areas = []
    for panel in panel_data:
        p1 = panel[1][0]
        p2 = panel[1][1]
        p3 = panel[1][2]
        p4 = panel[1][3]
        def triangle_area(p1, p2, p3):
            """Calculate the area of a triangle given three 3D points using the cross product."""
            v1 = np.array(p2) - np.array(p1)  # Vector from p1 to p2
            v2 = np.array(p3) - np.array(p1)  # Vector from p1 to p3
            cross_product = np.cross(v1, v2)  # Compute cross product
            return np.linalg.norm(cross_product) / 2  # Area is half the magnitude

        def quadrilateral_area_3d(p1, p2, p3, p4):
            """Calculate the area of a 3D quadrilateral by splitting it into two triangles."""
            area1 = triangle_area(p1, p2, p3)
            area2 = triangle_area(p1, p3, p4)
            return area1 + area2

        panel_areas.append(quadrilateral_area_3d(p1, p2, p3, p4))
    total_area = sum(panel_areas)

    return panel_data, total_area, panel_areas

def curvature(self):
    """
    Computes the curvature distribution for the entire aircraft using the plane dictionary
    and panel_data, which discretizes the plane into panels.

    Parameters:
        plane      : Dictionary with 'wing_sections', 'horizontal_stabilizer', and 'vertical_stabilizer'.
        panel_data : List of tuples (index, panel, quarter_chord_line, three_quarter_chord_midpoint, normal_vector).
        n          : Number of spanwise panels for the wing.
        m          : Number of chordwise panels.

    Returns:
        dz_dx_list : List of dz/dx values for all panels in panel_data.
    """
    plane = self.plane
    panel_data = self.panel_data
    def surface_dimensions(name):
        surface = self.discretization.get(name)
        if not surface:
            return 0, 0, 0
        span_count = max(0, len(surface['vertical_points']) - 1)
        chord_count = max(0, len(surface['horizontal_points']) - 1)
        return span_count, chord_count, span_count * chord_count

    n, m, num_wing_panels = surface_dimensions('wing')

    dz_dx_list = []
    if 'wing_sections' in plane:
        wing_panels = panel_data[:num_wing_panels]
        wing = compute_surface_curvature(plane['wing_sections'], wing_panels, n, m, is_symmetric=True)
        dz_dx_list = wing
    if 'horizontal_stabilizer' in plane:
        n_hs, m_hs, num_hs_panels = surface_dimensions('horizontal_stabilizer')
        hs_panels = panel_data[num_wing_panels:num_wing_panels + num_hs_panels]
        hs = compute_surface_curvature([plane['horizontal_stabilizer']], hs_panels, n_hs, m_hs, is_symmetric=True)
        dz_dx_list = np.hstack((dz_dx_list, hs))
    if 'vertical_stabilizer' in plane:
        n_vs, m_vs, num_vs_panels = surface_dimensions('vertical_stabilizer')
        vs_panels = panel_data[num_wing_panels + num_hs_panels : num_wing_panels + num_hs_panels + num_vs_panels]
        vs = compute_surface_curvature([plane['vertical_stabilizer']], vs_panels, n_vs, m_vs, is_symmetric=False, type='vs')
        dz_dx_list = np.hstack((dz_dx_list, vs))
    return dz_dx_list

def compute_surface_curvature(surface_data, panel_data, n, m, is_symmetric, type='wing'):
    """
    Computes curvature for a given surface (wing, horizontal, or vertical stabilizer).

    Parameters:
        surface_data : List of dictionaries with airfoil data (e.g., NACA_root, NACA_tip, span_fraction).
        panel_data   : List of panel data for the surface.
        n            : Number of spanwise panels for the surface.
        m            : Number of chordwise panels.
        is_symmetric : Boolean indicating if the surface is symmetric about the x-z plane.

    Returns:
        dz_dx_list   : List of dz/dx values for the surface panels.
    """
    if type == 'wing':
        total_span = panel_data[-1][1][3][0] * 2
        span_positions = np.linspace(0, total_span, n // 2)
        x_bounds = np.cumsum([0] + [sec['span_fraction'] for sec in surface_data])
        
        dz_dx = np.zeros((len(span_positions), m))
        dz_dx_total = None

        for sec_idx, sec in enumerate(surface_data):
            NACA_root = sec['NACA_root']
            NACA_tip = sec['NACA_tip']

            dz_root = np.array([])
            dz_tip = np.array([])

            span_idx = [i for i, x in enumerate(span_positions)
                        if x_bounds[sec_idx] <= x <= x_bounds[sec_idx + 1]]
            #print(f"Processing section {sec_idx + 1} with {len(span_idx)} points.")
            
            if not span_idx:
                continue

            for i, idx in enumerate(span_idx):
                x1, _, _ = panel_data[idx * m + int(n * m / 2)][1][0]
                x2, _, _ = panel_data[idx * m + int(n * m / 2)][1][3]
                #print(f"Processing span {idx}, coordinate: (x1:{x1}, x2:{x2}), position: {i + 1} of {len(span_idx)}.")

                for j in range(m):
                    y1 = j / m
                    y2 = (j + 1) / m

                    root = panel_data[span_idx[0] * m + int(n * m / 2)][1][0][0]
                    tip = panel_data[span_idx[-1] * m + int(n * m / 2)][1][3][0]

                    if np.isclose(x1, root, atol=1e-6):
                        intrados_root = y1
                        extrados_root = y2
                        #print(f"ROOT: Root: {root}, Tip: {tip}, x1: {x1}, intrados_root: {intrados_root}, extrados_root: {extrados_root}")
                        z1_root = naca_airfoil_dzdx(NACA_root, intrados_root)
                        z2_root = naca_airfoil_dzdx(NACA_root, extrados_root)
                        # Normalize by the chordwise step so this is a slope (dz/dx), not a raw height delta
                        dz_root_ = (z1_root - z2_root) / (y2 - y1)
                        dz_root = np.append(dz_root, dz_root_)

                    if np.isclose(x2, tip, atol=1e-6):
                        intrados_tip = y1
                        extrados_tip = y2
                        #print(f"TIP: Root: {root}, Tip: {tip}, x2: {x2}, intrados_tip: {intrados_tip}, extrados_tip: {extrados_tip}")
                        z1_tip = naca_airfoil_dzdx(NACA_tip, intrados_tip)
                        z2_tip = naca_airfoil_dzdx(NACA_tip, extrados_tip)
                        dz_tip_ = (z1_tip - z2_tip) / (y2 - y1)
                        dz_tip = np.append(dz_tip, dz_tip_)

            dz_dx = np.zeros((len(dz_tip), len(span_idx)))
            for i in range(len(dz_tip)):
                dz_dx[i] = np.linspace(dz_root[i], dz_tip[i], len(span_idx))

            if sec_idx == 0:
                dz_dx_total = dz_dx.copy()
            else:
                dz_dx_total = np.hstack((dz_dx_total, dz_dx))

        if is_symmetric:
            dz_dx_list = np.hstack((dz_dx_total[:, ::-1], dz_dx_total)).ravel('F').tolist()
        else:
            dz_dx_list = dz_dx_total.ravel('F').tolist()
    
    else: # Vertical stabilizer
        total_span = panel_data[-1][1][3][2]
        #print(f"Total span: {total_span}")
        span_positions = np.linspace(0, total_span, int(n))
        x_bounds = np.cumsum([0] + [sec['span_fraction'] for sec in surface_data])
        #print(f"Span positions: {span_positions}", f"X bounds: {x_bounds}")
        dz_dx = np.zeros((len(span_positions), m))
        dz_dx_total = None

        for sec_idx, sec in enumerate(surface_data):
            NACA_root = sec['NACA_root']
            NACA_tip = sec['NACA_tip']

            dz_root = np.array([])
            dz_tip = np.array([])

            span_idx = [i for i, x in enumerate(span_positions)
                        if x_bounds[sec_idx] <= x <= x_bounds[sec_idx + 1]]
            #print(f"Processing section {sec_idx + 1} with {len(span_idx)} points.")
            
            if not span_idx:
                continue

            for i, idx in enumerate(span_idx):
                #print(f" index: {idx * m}")
                x1, y1, z1 = panel_data[idx * m][1][0]
                x2, y2, z2 = panel_data[idx * m][1][3]
                #print(f"TIP: Processing span {idx}, coordinate: (x1:{x1}, y1:{y1}, z1:{z1}), position: {i + 1} of {len(span_idx)}.")
                #print(f"ROOT: Processing span {idx}, coordinate: (x2:{x2}, y2:{y2}, z2:{z2}), position: {i + 1} of {len(span_idx)}.")

                for j in range(m):
                    y1 = j / m
                    y2 = (j + 1) / m

                    root = panel_data[span_idx[0] * m][1][0][2]
                    tip = panel_data[span_idx[-1] * m][1][3][2]

                    if np.isclose(z1, root, atol=1e-6):
                        intrados_root = y1
                        extrados_root = y2
                        #print(f"ROOT: Root: {root}, Tip: {tip}, z1: {z1}, intrados_root: {intrados_root}, extrados_root: {extrados_root}")
                        z1_root = naca_airfoil_dzdx(NACA_root, intrados_root)
                        z2_root = naca_airfoil_dzdx(NACA_root, extrados_root)
                        # Normalize by the chordwise step so this is a slope (dz/dx), not a raw height delta
                        dz_root_ = (z1_root - z2_root) / (y2 - y1)
                        dz_root = np.append(dz_root, dz_root_)

                    elif np.isclose(z2, tip, atol=1e-6):
                        intrados_tip = y1
                        extrados_tip = y2
                        #print(f"TIP: Root: {root}, Tip: {tip}, z2: {z2}, intrados_tip: {intrados_tip}, extrados_tip: {extrados_tip}")
                        z1_tip = naca_airfoil_dzdx(NACA_tip, intrados_tip)
                        z2_tip = naca_airfoil_dzdx(NACA_tip, extrados_tip)
                        dz_tip_ = (z1_tip - z2_tip) / (y2 - y1)
                        dz_tip = np.append(dz_tip, dz_tip_)

            dz_dx = np.zeros((len(dz_tip), len(span_idx)))
            for i in range(len(dz_tip)):
                dz_dx[i] = np.linspace(dz_root[i], dz_tip[i], len(span_idx))

            if sec_idx == 0:
                dz_dx_total = dz_dx.copy()
            else:
                dz_dx_total = np.hstack((dz_dx_total, dz_dx))

        if is_symmetric:
            dz_dx_list = np.hstack((dz_dx_total[:, ::-1], dz_dx_total)).ravel('F').tolist()
        else:
            dz_dx_list = dz_dx_total.ravel('F').tolist()
        
    #print(f"dz_dx_list: {dz_dx_list}")
    return dz_dx_list

def interpolate_airfoil_z(x, y, z_root, z_tip, chord_root, chord_tip, x_tip_min, x_tip_max, sweep):
        """
        Interpolates z-height for a given x (spanwise position) and chordwise position (y).
        Parameters:
        - x: spanwise position (-1 to 1).
        - y: chordwise position (0 to 1).
        - z_root: Z-coordinates of the root airfoil (list or array).
        - z_tip: Z-coordinates of the tip airfoil (list or array).
        - chord_root: Root chord length.
        - chord_tip: Tip chord length.
        - y_root_min, y_root_max: Min and max chordwise positions for the root.
        - y_tip_min, y_tip_max: Min and max chordwise positions for the tip.
        - x_root_min, x_root_max: Spanwise bounds for the root.
        - x_tip_min, x_tip_max: Spanwise bounds for the tip.
        Returns:
        - Interpolated z-coordinate.
        """

        # Map x to span position [-1, 1]
        span_pos = 2 * (x - x_tip_min) / (x_tip_max - x_tip_min) - 1
        span_pos = np.clip(span_pos, -1, 1)

        # Interpolate the chord length at the current x position
        chord = chord_root + abs(span_pos) * (chord_tip - chord_root)

        # Normalize y to [0, 1] taking into account the sweep angle
        y = y - abs(x) * np.tan(sweep)
        y = np.clip(y, 0, 1)

        # Normalize y along the local chord to [0, 1] taking into account the sweep angle
        chord_fraction = y / chord

        # Ensure chord_fraction stays in bounds
        chord_fraction = np.clip(chord_fraction, 0, 1)

        # Interpolate z-height between root and tip airfoil
        z_root_interp = z_root[int(chord_fraction * (len(z_root) - 1))]
        z_tip_interp = z_tip[int(chord_fraction * (len(z_tip) - 1))]
        
        # Interpolate between root and tip z-coordinates
        z_interp = (1 - abs(span_pos)) * z_root_interp + abs(span_pos) * z_tip_interp
        return z_interp


######################################################################################################################
######################################################################################################################
# Geometry to STL functions ##########################################################################################
######################################################################################################################
######################################################################################################################

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

    from naca import naca_airfoil

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



######################################################################################################################
######################################################################################################################
# Plotting functions #################################################################################################
######################################################################################################################
######################################################################################################################

def plot_wing_discretization_3d(fig, panel_data):
    start_time = time.time()
    
    # Extract data from panel_data
    panels = [item[1] for item in panel_data]  # Panel vertices
    nodes = [item[2] for item in panel_data]   # 1/4 chord line nodes
    controls = [item[3] for item in panel_data]  # Control points
    normals = [item[4] for item in panel_data]   # Normal vectors (assuming item[4])

    fig = go.Figure()

    mesh_x, mesh_y, mesh_z = [], [], []
    mesh_i, mesh_j, mesh_k = [], [], []
    for panel in panels:
        vertex_offset = len(mesh_x)
        for x, y, z in panel:
            mesh_x.append(x)
            mesh_y.append(y)
            mesh_z.append(z)
        mesh_i.extend([vertex_offset, vertex_offset])
        mesh_j.extend([vertex_offset + 1, vertex_offset + 2])
        mesh_k.extend([vertex_offset + 2, vertex_offset + 3])

    fig.add_trace(go.Mesh3d(
        x=mesh_x, y=mesh_y, z=mesh_z,
        i=mesh_i, j=mesh_j, k=mesh_k,
        color='blue', opacity=0.1, showscale=False,
        name='Panels', showlegend=False
    ))

    edge_x, edge_y, edge_z = [], [], []
    for panel in panels:
        x, y, z = zip(*panel)
        edge_x.extend(list(x) + [x[0], None])  # Close the loop and separate with None
        edge_y.extend(list(y) + [y[0], None])
        edge_z.extend(list(z) + [z[0], None])
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        name='Panel edges', legendgroup='Panel edges', showlegend=True
    ))

    # 3. Control Points: Consolidated into one Scatter3d trace
    control_x = [control[0] for control in controls]
    control_y = [control[1] for control in controls]
    control_z = [control[2] for control in controls]
    fig.add_trace(go.Scatter3d(
        x=control_x, y=control_y, z=control_z,
        mode='markers', marker=dict(size=4, color='red'),
        name='Control points', legendgroup='Control points', showlegend=True,
        visible=False  # Initially hidden
    ))

    # 4. 1/4 Chord Lines: Consolidated into one Scatter3d trace
    line_x, line_y, line_z = [], [], []
    for line in nodes:
        x, y, z = zip(*line)
        line_x.extend(list(x) + [None])
        line_y.extend(list(y) + [None])
        line_z.extend(list(z) + [None])
    fig.add_trace(go.Scatter3d(
        x=line_x, y=line_y, z=line_z,
        mode='lines', line=dict(color='blue', width=5),
        name='1/4 Chord lines', legendgroup='1/4 Chord lines', showlegend=True,
        visible=False  # Initially hidden
    ))

    # 5. Normal Vectors: Consolidated lines and cones
    normal_line_x, normal_line_y, normal_line_z = [], [], []
    cone_x, cone_y, cone_z, cone_u, cone_v, cone_w = [], [], [], [], [], []
    for normal, control in zip(normals, controls):
        normal_vec, midpoint = normal  # Assuming normal contains [vector, midpoint]
        # Normal lines
        normal_line_x.extend([midpoint[0], midpoint[0] + normal_vec[0], None])
        normal_line_y.extend([midpoint[1], midpoint[1] + normal_vec[1], None])
        normal_line_z.extend([midpoint[2], midpoint[2] + normal_vec[2], None])
        # Normal cones
        cone_x.append(midpoint[0] + normal_vec[0])
        cone_y.append(midpoint[1] + normal_vec[1])
        cone_z.append(midpoint[2] + normal_vec[2])
        cone_u.append(normal_vec[0])
        cone_v.append(normal_vec[1])
        cone_w.append(normal_vec[2])

    fig.add_trace(go.Scatter3d(
        x=normal_line_x, y=normal_line_y, z=normal_line_z,
        mode='lines', line=dict(color='green', width=4),
        name='Normal vector', legendgroup='Normal vector', showlegend=False,
        visible=False  # Initially hidden
    ))

    fig.add_trace(go.Cone(
        x=cone_x, y=cone_y, z=cone_z,
        u=cone_u, v=cone_v, w=cone_w,
        sizemode="scaled", sizeref=1, anchor="tail",
        colorscale=[[0, 'green'], [1, 'green']], showscale=False,
        name='Normal vector', legendgroup='Normal vector', showlegend=True,
        visible=False  # Initially hidden
    ))

    # Define trace counts for visibility toggling
    num_panel_traces = 1             # Consolidated Mesh3d trace
    num_edge_traces = 1             # Consolidated edges
    num_control_traces = 1          # Consolidated control points
    num_node_traces = 1             # Consolidated 1/4 chord lines
    num_normal_traces = 2           # Normal lines + cones

    # Update layout with toggle button
    fig.update_layout(
        scene=dict(
            xaxis_title='y [m]', yaxis_title=' [m]', zaxis_title='z [m]',
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',
            camera =
                      {'eye':{'x':0,'y':1,'z':0}, 
                       'up': {'x':0,'y':0,'z':1}, 
                       'center': {'x':0,'y':0,'z':0}},
        ),
        title=dict(text='Wing Discretization'),
        legend=dict(x=0, y=1.02, orientation='h'),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1, xanchor="right",
                y=1.1, yanchor="top",
                showactive=False,
                buttons=[
                    dict(
                        label="Toggle normals and control points",
                        method="update",
                        args=[{
                            "visible": (
                                [True] * num_panel_traces +    # Panels
                                [True] * num_edge_traces +     # Edges
                                [False] * num_control_traces + # Control points
                                [False] * num_node_traces +    # 1/4 chord lines
                                [False] * num_normal_traces    # Normal vectors
                            )
                        }],
                        args2=[{
                            "visible": (
                                [True] * num_panel_traces +    # Panels
                                [True] * num_edge_traces +     # Edges
                                [True] * num_control_traces +  # Control points
                                [True] * num_node_traces +     # 1/4 chord lines
                                [True] * num_normal_traces     # Normal vectors
                            )
                        }]
                    )
                ]
            )
        ]
    )

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

    return fig

def plot_wing_discretization_2d(fig, panel_data):
    start_time = time.time()

    # Extract data from panel_data
    panels = [item[1] for item in panel_data]  # Panel vertices
    nodes = [item[2] for item in panel_data]   # 1/4 chord line nodes
    controls = [item[3] for item in panel_data]  # Control points

    # Initialize figure
    fig = go.Figure()

    # Store traces for each projection
    projections = ['xy', 'xz', 'yz']
    traces = {proj: [] for proj in projections}

    # Function to add traces for a given projection
    def add_2d_traces(proj):
        if proj == 'xy':
            x_idx, y_idx = 0, 1
            x_title, y_title = 'y [m]', 'x [m]'
        elif proj == 'xz':
            x_idx, y_idx = 0, 2
            x_title, y_title = 'y [m]', 'z [m]'
        elif proj == 'yz':
            x_idx, y_idx = 1, 2
            x_title, y_title = 'x [m]', 'z [m]'

        fill_px, fill_py = [], []
        edge_px, edge_py = [], []
        for panel in panels:
            px, py = [coord[x_idx] for coord in panel], [coord[y_idx] for coord in panel]
            fill_px.extend(list(px) + [px[0], None])
            fill_py.extend(list(py) + [py[0], None])
            edge_px.extend(list(px) + [px[0], None])
            edge_py.extend(list(py) + [py[0], None])

        traces[proj].append(go.Scattergl(
            x=fill_px, y=fill_py,
            fill='toself', fillcolor='rgba(0,0,255,0.1)',
            line=dict(color='blue'), mode='lines',
            name='Panel fill', showlegend=False, visible=(proj == 'xy')
        ))

        # Panel Edges
        traces[proj].append(go.Scattergl(
            x=edge_px, y=edge_py,
            mode='lines', line=dict(color='black', width=1),
            name='Panel edges', visible=(proj == 'xy')
        ))

        # Control Points
        control_px = [c[x_idx] for c in controls]
        control_py = [c[y_idx] for c in controls]
        traces[proj].append(go.Scattergl(
            x=control_px, y=control_py,
            mode='markers', marker=dict(size=3, color='red'),
            name='Control points', visible=(proj == 'xy')
        ))

        # 1/4 Chord Lines
        line_px, line_py = [], []
        for line in nodes:
            x, y, z = zip(*line)
            px, py = [coord[x_idx] for coord in line], [coord[y_idx] for coord in line]
            line_px.extend(list(px) + [None])
            line_py.extend(list(py) + [None])
        traces[proj].append(go.Scattergl(
            x=line_px, y=line_py,
            mode='lines', line=dict(color='blue', width=1),
            name='1/4 Chord lines', visible=(proj == 'xy')
        ))

        return x_title, y_title

    # Add traces for all projections
    axes_titles = {}
    for proj in projections:
        x_title, y_title = add_2d_traces(proj)
        axes_titles[proj] = (x_title, y_title)

    # Add all traces to the figure
    for proj in projections:
        for trace in traces[proj]:
            fig.add_trace(trace)

    # Create dropdown menu
    buttons = []
    for proj in projections:
        buttons.append(dict(
            label=proj.upper() + ' Projection',
            method='update',
            args=[
                {'visible': [proj == p for p in projections for _ in traces[p]]},
                {
                    'xaxis.title': axes_titles[proj][0],
                    'yaxis.title': axes_titles[proj][1]
                }
            ]
        ))

    # Update layout with dropdown
    fig.update_layout(
        title='Wing Discretization - 2D Projections',
        showlegend=True,
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                showactive=True,
                x=1.1,
                xanchor='right',
                y=1.1,
                yanchor='top'
            )
        ],
        xaxis=dict(title=axes_titles['xy'][0]),
        yaxis=dict(title=axes_titles['xy'][1]),
        legend=dict(x=0, y=1.02, orientation='h')
    )

    fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )

    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    return fig

def plot_wing_geometry(fig, wing_geometry, legend):

    if wing_geometry is None:
        raise ValueError("wing_geometry is None")

    required_keys = ['leading_edge', 'trailing_edge', 'quarter_chord']
    for key in required_keys:
        if key not in wing_geometry:
            raise ValueError(f"wing_geometry missing required key: {key}")
        if not isinstance(wing_geometry[key], (list, tuple)) or len(wing_geometry[key]) != 3:
            raise ValueError(f"Invalid format for {key}: expected tuple/list of 3 coordinate arrays")

    try:
        y_leading_edge, x_leading_edge, z_leading_edge = wing_geometry['leading_edge']
        y_trailing_edge, x_trailing_edge, z_trailing_edge = wing_geometry['trailing_edge']
        y_quarter_chord, x_quarter_chord, z_quarter_chord = wing_geometry['quarter_chord']

        # Plot the wing
        fig.add_trace(go.Scatter3d(
            x=y_leading_edge, y=x_leading_edge, z=z_leading_edge,
            mode='lines', line=dict(color='blue', width=2),
            name='Leading edge',
            legendgroup='Leading edge',
            showlegend=True if legend == "Wing Geometry" else False,  # Show legend only if not "Wing Geometry"
        ))
        fig.add_trace(go.Scatter3d(
            x=y_trailing_edge, y=x_trailing_edge, z=z_trailing_edge,
            mode='lines', line=dict(color='red', width=2),
            name='Trailing edge',
            legendgroup='Trailing edge',
            showlegend=True if legend == "Wing Geometry" else False,  # Show legend only if not "Wing Geometry"
        ))
        fig.add_trace(go.Scatter3d(
            x=y_quarter_chord, y=x_quarter_chord, z=z_quarter_chord,
            mode='lines', line=dict(color='green', width=2),
            name='Quarter-chord line',
            legendgroup='Quarter-chord line',
            showlegend=True if legend == "Wing Geometry" else False,  # Show legend only if not "Wing Geometry"
        ))
        # Set layout
        fig.update_layout(
            scene=dict(
            xaxis_title='x [m]',
            yaxis_title='y [m]',
            zaxis_title='z [m]',
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            aspectmode='data',  # Automatically adjust aspect ratio to fit data
            camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=0, z=2.5)
            )),
            title=dict(text='Wing Geometry'),
            legend=dict(x=0, y=1.02, orientation='h'),
        )

        return fig

    except Exception as e:
        raise ValueError(f"Error plotting wing geometry: {str(e)}")

def plot_wing_geometry_2d(fig, wing_geometry, legend):

    if wing_geometry is None:
        raise ValueError("wing_geometry is None")

    required_keys = ['leading_edge', 'trailing_edge', 'quarter_chord']
    for key in required_keys:
        if key not in wing_geometry:
            raise ValueError(f"wing_geometry missing required key: {key}")
        if not isinstance(wing_geometry[key], (list, tuple)) or len(wing_geometry[key]) != 3:
            raise ValueError(f"Invalid format for {key}: expected tuple/list of 3 coordinate arrays")

    try:
        x_leading_edge, y_leading_edge, _ = wing_geometry['leading_edge']
        x_trailing_edge, y_trailing_edge, _ = wing_geometry['trailing_edge']
        x_quarter_chord, y_quarter_chord, _ = wing_geometry['quarter_chord']
        
        # Plot leading edge
        fig.add_trace(go.Scatter(
            x=y_leading_edge, y=x_leading_edge,
            mode='lines', line=dict(width=2),
            name='Leading edge',
            legendgroup='Leading edge',
            showlegend=True if legend == "Wing Geometry" else False
        ))

        # Plot trailing edge
        fig.add_trace(go.Scatter(
            x=y_trailing_edge, y=x_trailing_edge,
            mode='lines', line=dict(width=2),
            name='Trailing edge',
            legendgroup='Trailing edge',
            showlegend=True if legend == "Wing Geometry" else False
        ))

        # Plot quarter chord
        fig.add_trace(go.Scatter(
            x=y_quarter_chord, y=x_quarter_chord,
            mode='lines', line=dict(width=2),
            name='Quarter-chord line',
            legendgroup='Quarter-chord line',
            showlegend=True if legend == "Wing Geometry" else False
        ))

        # Plot chord line at root (connecting leading edge to trailing edge)
        fig.add_trace(go.Scatter(
            x=[y_leading_edge[0], y_trailing_edge[0]],
            y=[x_leading_edge[0], x_trailing_edge[0]],
            mode='lines', line=dict(width=2),
            name='Root chord',
            legendgroup='Root chord',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[y_leading_edge[-1], y_trailing_edge[-1]],
            y=[x_leading_edge[-1], x_trailing_edge[-1]],
            mode='lines', line=dict(width=2),
            name='Tip chord',
            legendgroup='Tip chord',
            showlegend=False
        ))

        # Set layout
        fig.update_layout(
            xaxis_title='y [m]',
            yaxis_title='x [m]',
            margin=dict(l=50, r=50, t=50, b=50),  # Add 50-pixel margin on all sides
            title=dict(text='Wing Geometry (2D)'),
            legend=dict(x=0, y=1.02, orientation='h'),
            showlegend=True
        )

        fig.update_yaxes(
            scaleanchor="x",
            scaleratio=1
        )
        return fig

    except Exception as e:
        raise ValueError(f"Error plotting wing geometry: {str(e)}")


