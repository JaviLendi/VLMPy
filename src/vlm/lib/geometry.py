#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('../')  # Add parent directory to path for module imports

import time
import numpy as np
import plotly.graph_objects as go
from lib.naca import naca_airfoil_dzdx

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

        # Return geometry as a dictionary
        return {
            'leading_edge': (x_leading_edge, y_leading_edge, z_leading_edge),
            'trailing_edge': (x_trailing_edge, y_trailing_edge, z_trailing_edge),
            'quarter_chord': (x_quarter_chord, y_quarter_chord, z_quarter_chord)
        }

    wing_geometry = 0
    hs_geometry = 0
    vs_geometry = 0
    if 'wing_sections' in plane:
        # Calculate geometry for wing sections
        wing_geometry = calculate_section_geometry(plane['wing_sections'], self.alpha, is_symmetric=True, x_translate=0, z_translate=0)
    if 'horizontal_stabilizer' in plane:
        # Calculate geometry for horizontal stabilizer
        hs = plane['horizontal_stabilizer']
        hs_geometry = calculate_section_geometry([hs], hs['alpha'], True, hs['x_translate'], hs['z_translate'])
    if 'vertical_stabilizer' in plane:
        # Calculate geometry for vertical stabilizer
        vs = plane['vertical_stabilizer']
        vs_geometry = calculate_section_geometry([vs], vs['alpha'], False, vs['x_translate'], vs['z_translate'])

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
        
        # Interpolate along chord for each span position
        wing_horizontal_lines   = [x_leading * (1 - s) + x_trailing * s for s in chord_fractions]
        wing_z_coordinates      = [z_leading * (1 - s) + z_trailing * s for s in chord_fractions]
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
    n = self.n
    m = self.m
    num_wing_panels = n * m
    
    dz_dx_list = []
    if 'wing_sections' in plane:
        wing_panels = panel_data[:num_wing_panels]
        wing = compute_surface_curvature(plane['wing_sections'], wing_panels, n, m, is_symmetric=True)
        dz_dx_list = wing
    if 'horizontal_stabilizer' in plane:
        n_hs = self.n_hs
        m_hs = self.m_hs
        num_hs_panels = n_hs * m_hs
        hs_panels = panel_data[num_wing_panels:num_wing_panels + num_hs_panels]
        hs = compute_surface_curvature([plane['horizontal_stabilizer']], hs_panels, n_hs, m_hs, is_symmetric=True)
        dz_dx_list = np.hstack((dz_dx_list, hs))
    if 'vertical_stabilizer' in plane:
        n_vs = self.n_vs
        m_vs = self.m_vs
        num_vs_panels = n_vs * m_vs
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
                        dz_root_ = z1_root - z2_root
                        dz_root = np.append(dz_root, dz_root_)

                    elif np.isclose(x2, tip, atol=1e-6):
                        intrados_tip = y1
                        extrados_tip = y2
                        #print(f"TIP: Root: {root}, Tip: {tip}, x2: {x2}, intrados_tip: {intrados_tip}, extrados_tip: {extrados_tip}")
                        z1_tip = naca_airfoil_dzdx(NACA_tip, intrados_tip)
                        z2_tip = naca_airfoil_dzdx(NACA_tip, extrados_tip)
                        dz_tip_ = z1_tip - z2_tip
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
                        dz_root_ = z1_root - z2_root
                        dz_root = np.append(dz_root, dz_root_)

                    elif np.isclose(z2, tip, atol=1e-6):
                        intrados_tip = y1
                        extrados_tip = y2
                        #print(f"TIP: Root: {root}, Tip: {tip}, z2: {z2}, intrados_tip: {intrados_tip}, extrados_tip: {extrados_tip}")
                        z1_tip = naca_airfoil_dzdx(NACA_tip, intrados_tip)
                        z2_tip = naca_airfoil_dzdx(NACA_tip, extrados_tip)
                        dz_tip_ = z1_tip - z2_tip
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

    for panel in panels:
        x, y, z = zip(*panel)
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
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
    num_panel_traces = len(panels)  # Mesh3d traces
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

        for panel in panels:
            px, py = [coord[x_idx] for coord in panel], [coord[y_idx] for coord in panel]
            traces[proj].append(go.Scatter(
            x=list(px) + [px[0]], y=list(py) + [py[0]],
            fill='toself', fillcolor='rgba(0,0,255,0.1)', line=dict(color='blue'),
            mode='lines',  # Only lines, no markers
            name='Panel fill', showlegend=False, visible=(proj == 'xy')
            ))

        # Panel Edges
        edge_px, edge_py = [], []
        for panel in panels:
            px, py = [coord[x_idx] for coord in panel], [coord[y_idx] for coord in panel]
            edge_px.extend(list(px) + [px[0], None])
            edge_py.extend(list(py) + [py[0], None])
        traces[proj].append(go.Scatter(
            x=edge_px, y=edge_py,
            mode='lines', line=dict(color='black', width=1),
            name='Panel edges', visible=(proj == 'xy')
        ))

        # Control Points
        control_px = [c[x_idx] for c in controls]
        control_py = [c[y_idx] for c in controls]
        traces[proj].append(go.Scatter(
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
        traces[proj].append(go.Scatter(
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
            name='Borde de ataque',
            legendgroup='Borde de ataque',
            showlegend=True if legend == "Wing Geometry" else False,  # Show legend only if not "Wing Geometry"
        ))
        fig.add_trace(go.Scatter3d(
            x=y_trailing_edge, y=x_trailing_edge, z=z_trailing_edge,
            mode='lines', line=dict(color='red', width=2),
            name='Borde de salida',
            legendgroup='Borde de salida',
            showlegend=True if legend == "Wing Geometry" else False,  # Show legend only if not "Wing Geometry"
        ))
        fig.add_trace(go.Scatter3d(
            x=y_quarter_chord, y=x_quarter_chord, z=z_quarter_chord,
            mode='lines', line=dict(color='green', width=2),
            name='Cuerda 1/4',
            legendgroup='Cuerda 1/4',
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
            title=dict(text='Geometría del ala'),
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
            name='Borde de ataque',
            legendgroup='Borde de ataque',
            showlegend=True if legend == "Wing Geometry" else False
        ))

        # Plot trailing edge
        fig.add_trace(go.Scatter(
            x=y_trailing_edge, y=x_trailing_edge,
            mode='lines', line=dict(width=2),
            name='Borde de salida',
            legendgroup='Borde de salida',
            showlegend=True if legend == "Wing Geometry" else False
        ))

        # Plot quarter chord
        fig.add_trace(go.Scatter(
            x=y_quarter_chord, y=x_quarter_chord,
            mode='lines', line=dict(width=2),
            name='Cuerda 1/4',
            legendgroup='Cuerda 1/4',
            showlegend=True if legend == "Wing Geometry" else False
        ))

        # Plot chord line at root (connecting leading edge to trailing edge)
        fig.add_trace(go.Scatter(
            x=[y_leading_edge[0], y_trailing_edge[0]],
            y=[x_leading_edge[0], x_trailing_edge[0]],
            mode='lines', line=dict(width=2),
            name='Cuerda en la punta',
            legendgroup='Cuerda en la punta',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[y_leading_edge[-1], y_trailing_edge[-1]],
            y=[x_leading_edge[-1], x_trailing_edge[-1]],
            mode='lines', line=dict(width=2),
            name='Cuerda en la punta',
            legendgroup='Cuerda en la punta',
            showlegend=False
        ))

        # Set layout
        fig.update_layout(
            xaxis_title='y [m]',
            yaxis_title='x [m]',
            margin=dict(l=50, r=50, t=50, b=50),  # Add 50-pixel margin on all sides
            title=dict(text='Geometría del ala (2D)'),
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


