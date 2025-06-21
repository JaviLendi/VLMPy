#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import logging
import pickle
import plotly.graph_objects as go
import plotly.colors
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure local module import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('../')

from TFG.vlm.VLMPy.lib.geometry import *

#from julia import Main as jl
#jl.include("VLMPy\\lib\\julia\\vortex.jl")
#jl.include("VLMPy\\lib\\julia\\solve.jl")

class VLM:
    def __init__(self, plane, u, rho, alpha, beta, n, m):
        # Input parameters
        self.plane              = plane
        self.u                  = u
        self.rho                = rho
        self.alpha              = alpha
        self.beta               = beta
        self.n                  = n
        self.m                  = m 
        # Computed parameters
        self.wing_span          = 0
        self.total_wing_span    = 0
        self.wing_geometry      = None
        self.hs_geometry        = None
        self.vs_geometry        = None
        self.n_hs               = 0
        self.m_hs               = 0
        self.n_vs               = 0
        self.m_vs               = 0
        self.panel_data         = None
        self.wing_area          = 0
        self.panel_areas        = None
        self.discretization     = None
        self.dz_c               = None
        self.wing_span          = 0
        self.hs_span            = 0
        self.vs_span            = 0
        self.wing_span          = 0
        self.results            = None

    def save_and_load_plane_variables(self, filename='plane_variables.txt', option='save_and_load'):
        # Save and load plane variables to/from a text file
        def save_plane_variables(self, filename):
            with open(filename, 'w') as file:
                for key, value in self.plane.items():
                    file.write(f"{key}: {value}\n")
    
        @staticmethod
        def load_plane_variables(filename):
            plane = {}
            with open(filename, 'r') as file:
                for line in file:
                    key, value = line.strip().split(': ', 1)
                    plane[key] = eval(value)  # Use eval to convert string back to Python data types
            return plane
        
        if option == 'save_and_load':
            # Save plane variables to a text file
            save_plane_variables(self, filename)
            # Load plane variables from a text file
            self.plane = load_plane_variables(filename)
        elif option == 'load':
            # Load plane variables from a text file
            self.plane = load_plane_variables(filename)
        elif option == 'save':
            # Save plane variables to a text file
            save_plane_variables(self, filename)
        else:
            raise ValueError("Invalid option. Use 'save_and_load', 'load', or 'save'.")

    def save_state(self, filename='vlm_state.pkl'):
        """
        Save the entire VLM object state to a binary file.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_state(self, filename='vlm_state.pkl'):
        """
        Load the VLM object state from a binary file.
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)

    def save_plane(self, filename='plane.json'):
        """JSON-dump just the plane dict."""
        with open(filename, 'w') as f:
            json.dump(self.plane, f, indent=2)

    def load_plane(self, filename='plane.json'):
        """Load plane dict from JSON and assign to self.plane."""
        with open(filename, 'r') as f:
            self.plane = json.load(f)

    def calculate_geometry(self):
        # Get total wing span
        for section in self.plane['wing_sections']:
            self.wing_span = section['span_fraction']
            self.total_wing_span += self.wing_span

        # Wing geometry
        self.wing_geometry, self.hs_geometry, self.vs_geometry = calculate_geometry(self)

    def calculate_discretization(self):
        # Densidad de paneles del ala
        wing_panel_density = self.n / self.total_wing_span
        wing_panel_density_chord = self.m / self.plane['wing_sections'][0]['chord_root'] + 1

        # Calcular paneles para estabilizadores
        if 'horizontal_stabilizer' in self.plane:
            self.n_hs = int(np.round(wing_panel_density * self.plane['horizontal_stabilizer']['span_fraction']))
            self.m_hs = int(np.round(wing_panel_density_chord * self.plane['horizontal_stabilizer']['chord_root']))
        else:
            self.n_hs, self.m_hs = 0, 0

        if 'vertical_stabilizer' in self.plane:
            self.n_vs = int(np.round(wing_panel_density * self.plane['vertical_stabilizer']['span_fraction']))
            self.m_vs = int(np.round(wing_panel_density_chord * self.plane['vertical_stabilizer']['chord_root']))
        else:
            self.n_vs, self.m_vs = 0, 0

        # Generar puntos interpolados
        wing_vertical_points, wing_horizontal_points, wing_z_points = interpolate_wing_points(self, type="wing")
        self.discretization = {'wing': {'vertical_points': wing_vertical_points, 'horizontal_points': wing_horizontal_points, 'z_points': wing_z_points}}

        if 'horizontal_stabilizer' in self.plane:
            hs_vertical_points, hs_horizontal_points, hs_z_points = interpolate_wing_points(self, type="hs")
            self.discretization['horizontal_stabilizer'] = {'vertical_points': hs_vertical_points, 'horizontal_points': hs_horizontal_points, 'z_points': hs_z_points}

        if 'vertical_stabilizer' in self.plane:
            vs_vertical_points, vs_horizontal_points, vs_z_points = interpolate_wing_points(self, type="vs")
            self.discretization['vertical_stabilizer'] = {'vertical_points': vs_vertical_points, 'horizontal_points': vs_horizontal_points, 'z_points': vs_z_points}

        # Generar paneles
        self.panel_data, self.wing_area, self.panel_areas = generate_plane_panels(self)
        print(f"wing_area: {self.wing_area}")

        # Calcular curvatura
        self.dz_c = curvature(self)
        print(f"Length of dz_c: {len(self.dz_c)}")
        print(f"Length of panel_data: {len(self.panel_data)}")

        # Verificar consistencia
        if len(self.dz_c) != len(self.panel_data):
            raise ValueError(f"Inconsistencia: len(dz_c) = {len(self.dz_c)} != len(panel_data) = {len(self.panel_data)}")

    def calculate_wing_lift(self):
        # Calculate wing lift
        self.w_i, self.P_ij, self.gammas, self.lift_wing2, self.lift, self.lift_sum, self.CL, self.CL_locals, self.drag_wing2, self.drag, self.drag_sum, self.CD, self.CD_locals = plane(self)

        self.lift_wing = self.lift_sum['lift_wing']
        if 'horizontal_stabilizer' in self.plane:
            self.lift_hs = self.lift_sum['lift_hs']
        if 'vertical_stabilizer' in self.plane:
            self.lift_vs = self.lift_sum['lift_vs']

        self.drag_wing = self.drag_sum['drag_wing']
        if 'horizontal_stabilizer' in self.plane:
            self.drag_hs = self.drag_sum['drag_hs']
        if 'vertical_stabilizer' in self.plane:
            self.drag_vs = self.drag_sum['drag_vs']

        self.CL_locals_wing = self.CL_locals['CL_wing']
        if 'horizontal_stabilizer' in self.plane:
            self.CL_locals_hs = self.CL_locals['CL_hs']
        if 'vertical_stabilizer' in self.plane:
            self.CL_locals_vs = self.CL_locals['CL_vs']

        self.CD_locals_wing= self.CD_locals['CD_wing']
        if 'horizontal_stabilizer' in self.plane:
            self.CD_locals_hs = self.CD_locals['CD_hs']
        if 'vertical_stabilizer' in self.plane:
            self.CD_locals_vs = self.CD_locals['CD_vs']

    def compute_coefficients_vs_alpha(self, angles_deg):
        """
        Compute CL and CD for a range of angles of attack and return results.

        Parameters:
        - vlm: VLM object
        - angles_deg: List of angles of attack in degrees

        Returns:
        - results: Dict with 'angles_deg', 'CL', and 'CD' lists
        """
        angles_rad = np.radians(angles_deg)
        cl_values = []
        cd_values = []

        original_alpha = self.alpha

        for alpha in angles_rad:
            self.alpha = alpha
            self.calculate_geometry()
            self.calculate_discretization()
            self.calculate_wing_lift()
            cl_values.append(self.CL)
            cd_values.append(self.CD)

        self.alpha = original_alpha

        self.results = {
            'angles_deg': angles_deg,
            'CL': cl_values,
            'CD': cd_values
        }

    def run_vlm(self):
        self.calculate_geometry()
        self.calculate_discretization()
        self.calculate_wing_lift()

# Calculate the aerodynamic influence coefficient matrix
def calculate_P_ij_(panel_data, u, dz_c, alpha, beta):
    n_controls                          = len(panel_data)
    P_ij                                = np.zeros((n_controls, n_controls))
    P_ij_resis                          = np.zeros((n_controls, n_controls))
    w_i, normal_vector_, panel_length, u_   = calculate_w_i(panel_data, u, dz_c, alpha, beta)

    r1  = np.zeros((n_controls, n_controls, 3))
    r1_ = np.zeros((n_controls, n_controls, 3))
    r2  = np.zeros((n_controls, n_controls, 3))
    r2_ = np.zeros((n_controls, n_controls, 3))
    r0  = np.zeros((n_controls, n_controls, 3))

    for i, panel_i in enumerate(panel_data):
        xi, yi, zi = panel_i[3]
        for j, panel_j in enumerate(panel_data):
            xj, yj, zj = panel_j[2][0]
            xjf, yjf, zjf = panel_j[2][1]

            C           = np.array([yi, xi, zi])
            A           = np.array([yj, xj, zj])
            B           = np.array([yjf, xjf, zjf])

            # Calculate the distance vector from control point to the nodes
            r1[i, j]    = C - A
            r1_[i, j]   = A - C
            r2[i, j]    = C - B
            r2_[i, j]   = B - C
            r0[i, j]    = B - A

    r1   = np.ascontiguousarray(r1, dtype=np.float64)
    r2   = np.ascontiguousarray(r2, dtype=np.float64)
    r0   = np.ascontiguousarray(r0, dtype=np.float64)
    r1_  = np.ascontiguousarray(r1_, dtype=np.float64)
    r2_  = np.ascontiguousarray(r2_, dtype=np.float64)

    V_AB, V_AInf, V_BInf = jl.calculate_vortex_terms_batch(r1, r2, r0, r1_, r2_)

    for i in range(n_controls):
        normal_vector = normal_vector_[i]
        for j in range(n_controls):

            V_AB_   = V_AB[i, j, 1] * normal_vector[1] + V_AB[i, j, 2] * normal_vector[2]
            V_AInf_ = V_AInf[i, j, 1] * normal_vector[1] + V_AInf[i, j, 2] * normal_vector[2]
            V_BInf_ = V_BInf[i, j, 1] * normal_vector[1] + V_BInf[i, j, 2] * normal_vector[2]

            P_ij[i, j]          = V_AB_ + V_AInf_ + V_BInf_
            P_ij_resis[i, j]    = V_AInf_ + V_BInf_

    return P_ij, P_ij_resis, w_i, panel_length

def calculate_P_ij(panel_data, u, dz_c, alpha, beta):
    """
    Calculates the coefficient matrix P_ij for the VLM method.
    Inputs:
    - panel_data: List with the information of each panel
    Outputs:
    - P_ij: Coefficient matrix P_ij
    """
    n_controls = len(panel_data)
    P_ij = np.zeros((n_controls, n_controls))
    P_ij_resis = np.zeros((n_controls, n_controls))
    w_i, normal_vector_, panel_length, u_ = calculate_w_i(panel_data, u, dz_c, alpha, beta)

    for i, panel_i in enumerate(panel_data):
        xi, yi, zi = panel_i[3]  # Control point for panel i

        normal_vector = normal_vector_[i]

        for j, panel_j in enumerate(panel_data):
            xj, yj, zj = panel_j[2][0]  # Node 1 of panel j
            xjf, yjf, zjf = panel_j[2][1]  # Node 2 of panel j

            ## Vortex Head
            C = np.array([yi, xi, zi]) # Control point
            A = np.array([yj, xj, zj]) # Node 1
            B = np.array([yjf, xjf, zjf]) # Node 2

            # Vector AC, BC, AB
            r1  = C - A # Vector AC
            r1_ = A - C # Vector CA

            r2  = C - B # Vector BC
            r2_ = B - C # Vector CB

            r0  = B - A # Vector AB

            r1_norm = np.linalg.norm(r1) # Modulo de AC
            r2_norm = np.linalg.norm(r2) # Modulo de BC
            
            # V_AB
            cross_r1_r2 = np.cross(r1, r2) # Producto cruz entre AC y BC (vector normal al panel)
            cross_r1_r2_norm = np.linalg.norm(cross_r1_r2) # Módulo del vector normal

            r0r1 = np.dot(r0, r1) # Producto entre AB y AC
            r0r2 = np.dot(r0, r2) # Producto entre AB y BC
            
            psi = cross_r1_r2 / np.abs(cross_r1_r2_norm)**2 if cross_r1_r2_norm > 1e-12 else np.zeros(3) # Vector normal al panel dividido por el módulo al cuadrado
            omega = r0r1 / r1_norm - r0r2 / r2_norm if r1_norm > 1e-12 and r2_norm > 1e-12 else 0
            V_AB = np.dot(psi, omega) / (4 * np.pi)
            V_AB_ = V_AB[1] * normal_vector[1] + V_AB[2] * normal_vector[2]
            
            # Cálculo de V_AInf y V_BInf como vectores
            V_AInf = np.zeros(3)
            V_AInf[1] = (r1[2] / (r1[2]**2+r1_[1]**2)) * (1 + r1[0] / r1_norm) / (4 * np.pi) if r1_norm > 1e-6 and r1[2]**2 > 1e-6 else 0 # Componente en dirección j
            V_AInf[2] = (r1_[1] / (r1[2]**2+r1_[1]**2)) * (1 + r1[0] / r1_norm) / (4 * np.pi) if r1_norm > 1e-6 and r1_[1]**2 > 1e-6 else 0 # Componente en dirección k
            V_AInf_ = V_AInf[1] * normal_vector[1] + V_AInf[2] * normal_vector[2]

            V_BInf = np.zeros(3)
            V_BInf[1] = - (r2[2] / (r2[2]**2+r2_[1]**2)) * (1 + r2[0] / r2_norm) / (4 * np.pi) if r2_norm > 1e-6 and r2[2]**2 > 1e-6 else 0
            V_BInf[2] = - (r2_[1] / (r2[2]**2+r2_[1]**2)) * (1 + r2[0] / r2_norm) / (4 * np.pi) if r2_norm > 1e-6 and r2_[1]**2 > 1e-6 else 0
            V_BInf_ =  V_BInf[1] * normal_vector[1] + V_BInf[2] * normal_vector[2]

            # Construcción de la matriz de coeficientes ...
            P_ij[i, j] = V_AB_ + V_AInf_ + V_BInf_
            P_ij_resis[i, j] = V_AInf_ + V_BInf_
            
    return P_ij, P_ij_resis, w_i, panel_length, u_

def calculate_w_i(panel_data, u, dz_c, alpha, beta):
    n_controls = len(panel_data)
    w_i = np.zeros(n_controls)
    deltaZ = np.zeros(n_controls)
    normal_vector_ = np.array([np.zeros(3) for _ in range(n_controls)])
    panel_length = np.zeros(n_controls)

    # Verificar que dz_c tenga la misma longitud que panel_data
    if len(dz_c) != n_controls:
        raise ValueError(f"Error: len(dz_c) = {len(dz_c)} no coincide con len(panel_data) = {n_controls}")

    for i, panel_i in enumerate(panel_data):
        normal = panel_i[4][0]
        normal_ = np.array([normal[1], normal[0], normal[2]])

        # Calcular longitud del panel
        panel_length[i] = np.linalg.norm(np.array(panel_i[1][0]) - np.array(panel_i[1][3]))

        # Calcular curvatura
        if panel_length[i] != 0:  # Evitar división por cero
            deltaZ[i] = dz_c[i] / panel_length[i]
        else:
            deltaZ[i] = 0
        deltaP = np.atan(deltaZ[i])

        # Resto del código sin cambios
        cos_deltaP = np.cos(deltaP)
        sin_deltaP = np.sin(deltaP)
        A = np.array([
            [cos_deltaP, 0, -sin_deltaP],
            [0, 1, 0],
            [sin_deltaP, 0, cos_deltaP]])
        normal_vector_[i] = np.dot(normal_, A)
        normal_vector = normal_vector_[i]

        Talpha = np.array([
            [np.cos(alpha), 0, -np.sin(alpha)],
            [0, 1, 0],
            [np.sin(alpha), 0, np.cos(alpha)]])
        Tbeta = np.array([
            [np.cos(beta), np.sin(beta), 0],
            [-np.sin(beta), np.cos(beta), 0],
            [0, 0, 1]])
        u_ = np.dot(Tbeta, np.dot(Talpha, np.array([0, 1, 0]))) * u
        w_i[i] = -np.dot(np.array([u_[1], u_[0], u_[2]]), normal_vector)

    return w_i, normal_vector_, panel_length, u_

def plane(self):
    # 1. Extraer atributos 
    panel_data     = self.panel_data
    u, dz_c        = self.u, self.dz_c
    alpha, beta    = self.alpha, self.beta
    rho            = self.rho
    n, m           = self.n, self.m
    n_hs, m_hs     = self.n_hs, self.m_hs
    n_vs, m_vs     = self.n_vs, self.m_vs
    wing_area      = self.wing_area
    panel_areas    = self.panel_areas

    # 2. Resolver sistema P·γ = w
    P_ij, P_ij_resis, w_i, panel_lengths, self.u_ = calculate_P_ij(panel_data, u, dz_c, alpha, beta)
    gammas    = np.linalg.solve(P_ij, w_i)
    W_i       = P_ij_resis.dot(gammas)

    # 3. Parámetros aerodinámicos básicos
    q_total   = 0.5 * rho * u**2 * wing_area            # carga dinámica total del ala
    q_local   = 0.5 * rho * u**2                         # carga dinámica local (por panel)

    # 4. Separar subconjuntos de paneles
    num_wing_panels = n * m
    num_hs_panels   = n_hs * m_hs
    num_vs_panels   = n_vs * m_vs

    # Áreas por sub-superficie
    A_wing = panel_areas[:num_wing_panels]
    A_hs   = panel_areas[num_wing_panels:num_wing_panels + num_hs_panels]
    A_vs   = panel_areas[num_wing_panels + num_vs_panels:]

    # 5. Calcular lift y drag elementales por panel
    lift_per_panel = rho * u * gammas * panel_lengths    
    drag_per_panel = - rho * W_i * gammas * panel_lengths

    # 6. Función auxiliar para sumar y obtener coeficientes locales
    def coefs(forces, areas, n_sub, m_sub):
        """
        Dado un vector forces (len = n_sub*m_sub), y vector areas (mismo len),
        lo redimensiona en matriz (n_sub, m_sub), suma en cada fila,
        devuelve: sumas_por_fila, suma_total, CL_local_por_fila.
        """
        mat_f = forces.reshape(n_sub, m_sub)
        mat_A = areas.reshape(n_sub, m_sub)

        sum_fila   = mat_f.sum(axis=1)       # array longitud n_sub
        total      = sum_fila.sum()          # escalar

        area_strip = mat_A.sum(axis=1)       # array longitud n_sub
        CL_local   = sum_fila / (q_local * area_strip)

        return sum_fila, total, CL_local

    # 7. Wing: lift
    lift_wing_vec = lift_per_panel[:num_wing_panels]
    lift_sum_wing, lift_tot_wing, CL_locals_wing = coefs(lift_wing_vec, np.array(A_wing), n, m)

    # 8. HS: lift
    if n_hs != 0:
        lift_hs_vec = lift_per_panel[num_wing_panels:num_wing_panels + num_hs_panels]
        lift_sum_hs, lift_tot_hs, CL_locals_hs = coefs(lift_hs_vec, np.array(A_hs), n_hs, m_hs)
    else:
        lift_hs_vec = np.zeros(num_hs_panels)
        lift_sum_hs = np.zeros(n_hs)
        lift_tot_hs = 0.0
        CL_locals_hs = np.zeros(n_hs)

    # 9. VS: lift (mismos n_hs, m_hs)
    if n_vs != 0:
        lift_vs_vec = lift_per_panel[num_wing_panels + num_hs_panels:]
        lift_sum_vs, lift_tot_vs, CL_locals_vs = coefs(lift_vs_vec, np.array(A_vs), n_vs, m_vs)
    else:
        lift_vs_vec = np.zeros(num_vs_panels)
        lift_sum_vs = np.zeros(n_vs)
        lift_tot_vs = 0.0
        CL_locals_vs = np.zeros(n_vs)

    # 10. Resumen global de lift
    lift_sum_dict  = {'lift_wing': lift_sum_wing, 'lift_hs': lift_sum_hs, 'lift_vs': lift_sum_vs}
    lift_total_all = lift_tot_wing + lift_tot_hs + lift_tot_vs
    CL_total       = lift_total_all / q_total
    CL_locals_dict = {'CL_wing': CL_locals_wing, 'CL_hs': CL_locals_hs, 'CL_vs': CL_locals_vs}

    # 11. Wing: drag
    drag_wing_vec = drag_per_panel[:num_wing_panels]
    drag_sum_wing, drag_tot_wing, _ = coefs(drag_wing_vec, np.array(A_wing), n, m)

    # 12. HS: drag
    if n_hs != 0:
        drag_hs_vec = drag_per_panel[num_wing_panels:num_wing_panels + num_hs_panels]
        drag_sum_hs, drag_tot_hs, _ = coefs(drag_hs_vec, np.array(A_hs), n_hs, m_hs)
    else:
        drag_hs_vec = np.zeros(num_hs_panels)
        drag_sum_hs = np.zeros(n_hs)
        drag_tot_hs = 0.0

    # 13. VS: drag
    if n_vs != 0:
        drag_vs_vec = drag_per_panel[num_wing_panels + num_hs_panels:]
        drag_sum_vs, drag_tot_vs, _ = coefs(drag_vs_vec, np.array(A_vs), n_hs, m_hs)
    else:
        drag_vs_vec = np.zeros(num_vs_panels)
        drag_sum_vs = np.zeros(n_vs)
        drag_tot_vs = 0.0

    # 14. Resumen global de drag
    drag_sum_dict  = {'drag_wing': drag_sum_wing, 'drag_hs': drag_sum_hs, 'drag_vs': drag_sum_vs}
    drag_total_all = drag_tot_wing + drag_tot_hs + drag_tot_vs
    CD_total       = drag_total_all / q_total

    # 15. Coeficientes de drag locales (por strip) para cada sub-superficie
    #     CD_local_i = drag_strip_i / (q_local * area_strip_i)
    #    (Reutilizamos area_strip calculado en la función auxiliar)
    _, _, CD_locals_wing = coefs(drag_wing_vec, np.array(A_wing), n, m)

    if n_hs != 0:
        _, _, CD_locals_hs   = coefs(drag_hs_vec, np.array(A_hs), n_hs, m_hs)
    else:
        CD_locals_hs = np.zeros(n_hs)
    if n_vs != 0:
        _, _, CD_locals_vs   = coefs(drag_vs_vec, np.array(A_vs), n_hs, m_hs)
    else:
        CD_locals_vs = np.zeros(n_vs)
    CD_locals_dict = {
        'CD_wing': CD_locals_wing,
        'CD_hs':   CD_locals_hs,
        'CD_vs':   CD_locals_vs
    }

    # 16. Devolver todos los resultados necesarios
    return (
        w_i,
        P_ij,
        gammas,
        lift_wing_vec,             # vectores individuales si se necesitan
        lift_total_all,
        lift_sum_dict,
        CL_total,
        CL_locals_dict,
        drag_wing_vec,
        drag_total_all,
        drag_sum_dict,
        CD_total,
        CD_locals_dict
    )

############################################################################################################
############################################################################################################
# Plotting functions
############################################################################################################
############################################################################################################

def plot_wing_heatmap(fig, panel_data, gammas, title='Wing Heatmap', legend='Gamma Value', 
                      colorscale='Viridis', opacity=0.8, show_edges=True):
    """
    Plot a 3D heatmap of gamma values on a wing planform using Plotly.
    Each panel is colored according to its circulation (gamma) value, with optional edge lines.

    Parameters:
    - fig: Plotly Figure object to update (modified in-place)
    - panel_data: List of panels, where each panel[1] contains corner coordinates [[x, y, z], ...]
    - gammas: Array of gamma values corresponding to each panel
    - title: Title of the plot (default: 'Wing Heatmap')
    - legend: Label for the colorbar (default: 'Gamma Value')
    - colorscale: Plotly colorscale for heatmap (default: 'Viridis')
    - opacity: Opacity of panels (default: 0.8)
    - show_edges: Whether to draw panel edges (default: True)

    Returns:
    - fig: Updated Plotly Figure object

    Raises:
    - ValueError: If panel_data or gammas are empty or mismatched
    """

    # Convert gammas to numpy array and compute min/max
    gammas = np.array(gammas)
    vmin, vmax = np.min(gammas), np.max(gammas)

    # Collect unique vertices and assign global indices
    vertex_to_index = {}
    global_vertices = []
    index = 0
    for panel in panel_data:
        for vertex in panel[1]:
            vertex_tuple = tuple(vertex)
            if vertex_tuple not in vertex_to_index:
                vertex_to_index[vertex_tuple] = index
                global_vertices.append(vertex)
                index += 1

    # Build triangle indices (i, j, k) and intensity for all panels
    i_all, j_all, k_all = [], [], []
    intensity = []
    for i, panel in enumerate(panel_data):
        # Map panel vertices to global indices
        panel_indices = [vertex_to_index[tuple(vertex)] for vertex in panel[1]]
        # Define two triangles: 0-1-2 and 0-2-3
        i_all.extend([panel_indices[0], panel_indices[0]])
        j_all.extend([panel_indices[1], panel_indices[2]])
        k_all.extend([panel_indices[2], panel_indices[3]])
        # Assign gamma value to both triangles
        intensity.extend([gammas[i], gammas[i]])

    # Extract global x, y, z coordinates
    x = [v[0] for v in global_vertices]
    y = [v[1] for v in global_vertices]
    z = [v[2] for v in global_vertices]

    # Create single Mesh3d trace for all panels
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i_all, j=j_all, k=k_all,
        intensity=intensity,
        intensitymode='cell',  # Color per triangle
        colorscale=colorscale,
        cmin=vmin,
        cmax=vmax,
        colorbar=dict(title=legend, titleside='right', thickness=20),
        opacity=opacity,
        flatshading=False,
        name='Wing Heatmap',
        hovertemplate='Value: %{intensity:.4f}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}'
    )

    traces = [mesh]

    # Add edges as a single Scatter3d trace if requested
    if show_edges:
        edge_x, edge_y, edge_z = [], [], []
        for panel in panel_data:
            vertices = np.array(panel[1])
            x_panel, y_panel, z_panel = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            # Close the loop and add None separator
            edge_x.extend(list(x_panel) + [x_panel[0], None])
            edge_y.extend(list(y_panel) + [y_panel[0], None])
            edge_z.extend(list(z_panel) + [z_panel[0], None])
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='black', width=2, dash='solid'),
            opacity=0.2,
            hoverinfo='skip',
            showlegend=False
        )
        traces.append(edge_trace)

    # Set axis ranges
    all_vertices = np.vstack([panel[1] for panel in panel_data])
    x_range = [np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])]
    y_range = [np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])]
    z_range = [np.min(all_vertices[:, 2]), np.max(all_vertices[:, 2])]

    # Define layout
    layout = go.Layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis=dict(title='y [m]', range=x_range),
            yaxis=dict(title='x [m]', range=y_range),
            zaxis=dict(title='z [m]', range=z_range),
            aspectmode='data',  # Equal aspect ratio
            camera=dict(
                up=dict(x=1, y=0, z=1),
                eye=dict(x=0, y=0, z=(z_range[1] - z_range[0]))  # Top-down view
            )
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=50, b=0)  # Tight margins
    )

    # Update the figure and return
    fig.update(data=traces, layout=layout)
    return fig

def plot_wing_heatmap_2d(fig, panel_data, gammas, title='Wing Heatmap', legend='Gamma Value', u_vec=None, 
                         colorscale='Viridis', opacity=0.8, show_edges=True):
    """
    Plot a 2D heatmap of gamma values on a wing planform in the XY plane using Plotly.
    Each panel is colored according to its circulation (gamma) value, with optional edge lines.

    Parameters:
    - fig: Plotly Figure object to update (modified in-place)
    - panel_data: List of panels, where each panel[1] contains corner coordinates [[x, y, z], ...]
    - gammas: Array of gamma values corresponding to each panel
    - title: Title of the plot (default: 'Wing Heatmap')
    - legend: Label for the colorbar (default: 'Gamma Value')
    - colorscale: Plotly colorscale for heatmap (default: 'Viridis')
    - opacity: Opacity parameter (default: 0.8, not used in 2D)
    - show_edges: Whether to draw panel edges (default: True)

    Returns:
    - fig: Updated Plotly Figure object

    Raises:
    - ValueError: If panel_data or gammas are empty or mismatched
    """
    # Convert gammas to numpy array and compute min/max
    gammas = np.array(gammas)
    vmin, vmax = np.min(gammas), np.max(gammas)
    if vmin == vmax:
        normalized_gammas = [0.5] * len(gammas)
    else:
        normalized_gammas = (gammas - vmin) / (vmax - vmin)

    # Get colors from colorscale
    colors = plotly.colors.sample_colorscale(colorscale, normalized_gammas)

    # Create dummy heatmap for colorbar
    dummy_z = [[vmin, vmax]]
    dummy_x = [0, 0.01]
    dummy_y = [0, 0.01]
    dummy_heatmap = go.Heatmap(
        x=dummy_x,
        y=dummy_y,
        z=dummy_z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=legend, titleside='right', thickness=20),
        visible=True
    )
    fig.add_trace(dummy_heatmap)

    # Create traces for each panel
    for i, panel in enumerate(panel_data):
        vertices = panel[1]
        x_panel = [vertex[0] for vertex in vertices] + [vertices[0][0]]
        y_panel = [vertex[1] for vertex in vertices] + [vertices[0][1]]
        trace = go.Scatter(
            x=x_panel,
            y=y_panel,
            mode='lines',
            fill='toself',
            fillcolor=colors[i],
            line=dict(width=1 if show_edges else 0, color='black'),
            hoverinfo='text',
            text=f'Value: {gammas[i]:.4f}',
            showlegend=False
        )
        fig.add_trace(trace)

    # Plot the velocity vector u if provided
    if u_vec is not None and u_vec[0] > 0.1:
        u_x, u_y = u_vec[0], u_vec[1]
        # Place the arrow at the origin (0, 0)
        x0, y0 = 0.0, 0.0
        # Scale the arrow for visibility
        all_vertices = np.vstack([panel[1] for panel in panel_data])
        arrow_scale = 0.1 * max(np.ptp(all_vertices[:, 0]), np.ptp(all_vertices[:, 1])) / (np.linalg.norm([u_x, u_y]) + 1e-8)
        dx = u_x * arrow_scale
        dy = u_y * arrow_scale
        # Add an annotation arrow to represent the velocity vector
        fig.add_annotation(
            x=x0,
            y=y0,
            ax=x0 - dx,
            ay=y0 - dy,
            xref='x',
            yref='y',
            axref='x',
            ayref='y',
            text='u',
            showarrow=True,
            arrowhead=3,
            arrowsize=2,
            arrowwidth=2,
            arrowcolor='blue',
            font=dict(color='blue', size=14),
            align='center',
        )
        
    # Set axis ranges
    all_vertices = np.vstack([panel[1] for panel in panel_data])
    x_range = [np.min(all_vertices[:, 0]), np.max(all_vertices[:, 0])]
    y_range = [np.min(all_vertices[:, 1]), np.max(all_vertices[:, 1])]

    # Define layout
    layout = go.Layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(title='y [m]', range=x_range),
        yaxis=dict(title='x [m]', range=y_range, scaleanchor='x', scaleratio=1),
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    # Update the figure layout
    fig.update_layout(layout)
    return fig

def plot_wing_lift(fig, vlm, lift, title='Wing Distribution', legend='Lift [N]'):
    """
    Plot the lift distribution on the wing using Plotly.

    Parameters:
    - panel_data: List of panels
    - lift: Array of lift values corresponding to each panel
    - n, m: Number of panels along spanwise and chordwise directions for the wing
    - n_hs, m_hs: Number of panels for the horizontal stabilizer
    - title: Title of the plot (default: 'Wing Lift Distribution')
    - legend: Label for the y-axis (default: 'Lift [N]')
    """
    # Define panel counts for each surface
    num_wing_panels = vlm.n * vlm.m
    num_hs_panels   = vlm.n_hs * vlm.m_hs
    num_vs_panels  = vlm.n_vs * vlm.m_vs

    # Split panel_data into surface-specific lists
    wing_panels     = vlm.panel_data[:num_wing_panels]
    hs_panels       = vlm.panel_data[num_wing_panels:num_wing_panels + num_hs_panels]
    vs_panels       = vlm.panel_data[num_wing_panels + num_hs_panels:]

    if title == 'Wing Lift Distribution' or title == 'Wing Drag Distribution' or title == 'Wing CL Distribution' or title == 'Wing CD Distribution':
        x_controls = [control[3][0] for control in wing_panels]
    elif title == 'HS Lift Distribution' or title == 'HS Drag Distribution':
        x_controls = [control[3][0] for control in hs_panels]
    elif title == 'VS Lift Distribution' or title == 'VS Drag Distribution':
        x_controls = [control[3][0] for control in vs_panels]
    else:
        raise ValueError("Invalid title for lift distribution plot.")
    x_controls = np.array(x_controls)

    x_controls_unique = []
    for x in x_controls:
        if x not in x_controls_unique:
            x_controls_unique.append(x)

    mid_index = int(len(lift) / 2)
    lift[mid_index:] = lift[mid_index:][::1]

    # Create the lift distribution plot
    fig.add_trace(go.Scatter(
        x=x_controls_unique,
        y=lift,
        mode='lines+markers',
        name=legend
    ))

    # Define layout
    fig.update_layout(
        title=title,
        xaxis_title='y [m]',
        yaxis_title=legend,
        autosize=True, 
    )

    return fig

def plot_distribution_all(fig, vlm, quantity='CL'):
    """
    Plot the distribution for wing, horizontal stabilizer (HS), and vertical stabilizer (VS) on the same figure.

    Parameters:
    - fig: Plotly Figure object to update (modified in-place)
    - vlm: VLM object with computed distributions
    - quantity: 'Lift', 'Drag', 'CL', or 'CD'

    Returns:
    - fig: Updated Plotly Figure object
    """
    quantity = quantity.upper()
    if quantity not in ['LIFT', 'DRAG', 'CL', 'CD']:
        raise ValueError("Invalid quantity. Use 'Lift', 'Drag', 'CL', or 'CD'.")

    # Set labels and attribute names
    if quantity == 'LIFT':
        attr_wing = getattr(vlm, 'lift_wing', None)
        attr_hs = getattr(vlm, 'lift_hs', None)
        attr_vs = getattr(vlm, 'lift_vs', None)
        y_label = 'Lift [N]'
        plot_label = 'Lift'
    elif quantity == 'DRAG':
        attr_wing = getattr(vlm, 'drag_wing', None)
        attr_hs = getattr(vlm, 'drag_hs', None)
        attr_vs = getattr(vlm, 'drag_vs', None)
        y_label = 'Drag [N]'
        plot_label = 'Drag'
    elif quantity == 'CL':
        attr_wing = getattr(vlm, 'CL_locals_wing', None)
        attr_hs = getattr(vlm, 'CL_locals_hs', None)
        attr_vs = getattr(vlm, 'CL_locals_vs', None)
        y_label = 'CL'
        plot_label = 'CL'
    elif quantity == 'CD':
        attr_wing = getattr(vlm, 'CD_locals_wing', None)
        attr_hs = getattr(vlm, 'CD_locals_hs', None)
        attr_vs = getattr(vlm, 'CD_locals_vs', None)
        y_label = 'CD'
        plot_label = 'CD'

    # Wing
    num_wing_panels = vlm.n * vlm.m
    if attr_wing is not None:
        wing_panels = vlm.panel_data[:num_wing_panels]
        x_controls = np.array([panel[3][0] for panel in wing_panels])
        values = np.array(attr_wing)
        if values.shape[0] == vlm.n:
            # Already spanwise
            x_span = x_controls[::vlm.m]
            fig.add_trace(go.Scatter(
                x=x_span,
                y=values,
                mode='lines+markers',
                name=f'Wing {plot_label}',
                line=dict(color='blue')
            ))
        else:
            # Panelwise, need to sum chordwise
            values = values.reshape(vlm.n, vlm.m).sum(axis=1)
            x_span = x_controls[::vlm.m]
            fig.add_trace(go.Scatter(
                x=x_span,
                y=values,
                mode='lines+markers',
                name=f'Wing {plot_label}',
                line=dict(color='blue')
            ))

    # HS
    num_hs_panels = vlm.n_hs * vlm.m_hs
    if attr_hs is not None and vlm.n_hs > 0 and vlm.m_hs > 0:
        start = num_wing_panels
        hs_panels = vlm.panel_data[start:start + num_hs_panels]
        x_controls = np.array([panel[3][0] for panel in hs_panels])
        values = np.array(attr_hs)
        if values.shape[0] == vlm.n_hs:
            x_span = x_controls[::vlm.m_hs]
            fig.add_trace(go.Scatter(
                x=x_span,
                y=values,
                mode='lines+markers',
                name=f'HS {plot_label}',
                line=dict(color='green')
            ))
        else:
            values = values.reshape(vlm.n_hs, vlm.m_hs).sum(axis=1)
            x_span = x_controls[::vlm.m_hs]
            fig.add_trace(go.Scatter(
                x=x_span,
                y=values,
                mode='lines+markers',
                name=f'HS {plot_label}',
                line=dict(color='green')
            ))

    # VS
    num_vs_panels = vlm.n_vs * vlm.m_vs
    if attr_vs is not None and vlm.n_vs > 0 and vlm.m_vs > 0:
        start = num_wing_panels + num_hs_panels
        vs_panels = vlm.panel_data[start:start + num_vs_panels]
        x_controls = np.array([panel[3][0] for panel in vs_panels])
        values = np.array(attr_vs)
        if values.shape[0] == vlm.n_vs:
            x_span = x_controls[::vlm.m_vs]
            fig.add_trace(go.Scatter(
                x=x_span,
                y=values,
                mode='lines+markers',
                name=f'VS {plot_label}',
                line=dict(color='red')
            ))
        else:
            values = values.reshape(vlm.n_vs, vlm.m_vs).sum(axis=1)
            x_span = x_controls[::vlm.m_vs]
            fig.add_trace(go.Scatter(
                x=x_span,
                y=values,
                mode='lines+markers',
                name=f'VS {plot_label}',
                line=dict(color='red')
            ))

    fig.update_layout(
        title=f'{plot_label} Distribution: Wing, HS, VS',
        xaxis_title='y [m]',
        yaxis_title=y_label,
        autosize=True,
        showlegend=True
    )
    return fig

def plot_coefficient_vs_alpha(fig, self, coefficient='CL', title=None):
    """
    Plot CL or CD vs angle of attack using precomputed results.
    Calculates and displays the slope (pendiente) of the line (in radians).

    Parameters:
    - coefficient: 'CL' or 'CD'
    - title: Optional title for the plot

    Returns:
    - fig: Plotly Figure object
    """
    if coefficient not in self.results:
        raise ValueError(f"'{coefficient}' not found in results.")

    x_deg = np.array(self.results['angles_deg'])
    y = np.array(self.results[coefficient])
    x_rad = np.radians(x_deg)

    # Calculate the slope (pendiente) using linear regression in radians
    slope_rad, intercept = np.polyfit(x_rad, y, 1)

    if title is None:
        title = f"{'Lift' if coefficient == 'CL' else 'Drag'} Coefficient vs Angle of Attack"

    fig.add_trace(go.Scatter(
        x=x_deg,
        y=y,
        mode='lines+markers',
        name=coefficient,
        line=dict(color='blue'),
        marker=dict(size=8)
    ))

    # Annotate the slope value (in radians) on the plot
    fig.add_annotation(
        xref='paper', yref='paper',
        x=0.05, y=0.95,
        text=f"Slope (rad): {slope_rad:.4f}",
        showarrow=False,
        font=dict(size=14, color='red'),
        bgcolor='white'
    )

    fig.update_layout(
        title=title,
        xaxis_title='Angle of Attack (degrees)',
        yaxis_title=f'{coefficient} Coefficient',
        autosize=True,
        showlegend=True
    )

    return fig

def plot_distribution(vlm, n_section, quantity='lift'):
    """
    Plot the distribution (lift or drag) along the chord for a given spanwise section.

    Parameters:
    - vlm: VLM object
    - n_section: Index of the spanwise section (0-based)
    - quantity: 'lift' or 'drag' (default: 'lift')

    Returns:
    - fig: Plotly Figure object
    """
    if n_section < 0 or n_section >= vlm.n:
        raise ValueError("n_section is out of range")

    start_idx = n_section * vlm.m
    end_idx = start_idx + vlm.m

    if quantity == 'lift':
        values = vlm.lift_wing2[start_idx:end_idx]
        y_label = 'Lift [N]'
        plot_title = f'Lift distribution (Section n={n_section})'
        color = 'blue'
    elif quantity == 'drag':
        values = vlm.drag_wing2[start_idx:end_idx]
        y_label = 'Drag [N]'
        plot_title = f'Drag distribution (Section n={n_section})'
        color = 'red'
    else:
        raise ValueError("quantity must be 'lift' or 'drag'")

    panels_section = vlm.panel_data[start_idx:end_idx]
    x_coords = [panel[3][1] for panel in panels_section]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=values,
        mode='lines+markers',
        name=quantity.capitalize(),
        line=dict(color=color),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=plot_title,
        xaxis_title='Chordwise position (x) [m]',
        yaxis_title=y_label,
        showlegend=True,
        template='plotly_white'
    )

    return fig

def plot_CL_CD(fig, vlm, title='CL vs CD of the Wing'):
    """
    Plot the CL/CD ratio along the span for the specified surface.

    Parameters:
    - fig: Plotly Figure object
    - vlm: VLM object
    - title: Title of the plot

    Returns:
    - fig: Plotly Figure object
    """
    if title == 'CL vs CD of the Wing':
        CL_local = vlm.CL_locals_wing
        CD_local = vlm.CD_locals_wing
        n = vlm.n
        label = 'Wing'
    elif title == 'CL vs CD of the HS':
        CL_local = getattr(vlm, 'CL_locals_hs', None)
        CD_local = getattr(vlm, 'CD_locals_hs', None)
        n = vlm.n_hs
        label = 'HS'
    elif title == 'CL vs CD of the VS':
        CL_local = getattr(vlm, 'CL_locals_vs', None)
        CD_local = getattr(vlm, 'CD_locals_vs', None)
        n = vlm.n_vs
        label = 'VS'
    else:
        raise ValueError("Title must be 'wing', 'hs', or 'vs'")

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        CL_over_CD = np.where(CD_local != 0, CL_local / CD_local, 0)

    # Get spanwise y-coordinates (control points)
    if title == 'CL vs CD of the Wing':
        panels = vlm.panel_data[:vlm.n * vlm.m]
    elif title == 'CL vs CD of the HS':
        start = vlm.n * vlm.m
        panels = vlm.panel_data[start:start + vlm.n_hs * vlm.m_hs]
    elif title == 'CL vs CD of the VS':
        start = vlm.n * vlm.m + vlm.n_hs * vlm.m_hs
        panels = vlm.panel_data[start:start + vlm.n_vs * vlm.m_vs]
    # Take the y-coordinate of the control point for each spanwise strip
    y_span = []
    for i in range(n):
        idx = i * (len(panels) // n)
        y_span.append(panels[idx][3][0])

    fig.add_trace(go.Scatter(
        x=y_span,
        y=CL_over_CD,
        mode='lines+markers',
        name=f'CL/CD ({label})',
        line=dict(color='purple'),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Spanwise Position y [m]',
        yaxis_title='CL/CD',
        showlegend=True,
        template='plotly_white'
    )

    return fig