import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Calculate wing geometry
def calculate_wing_geometry(chord_root, chord_tip, wing_span, sweep):
    """
    Calculates the coordinates of the leading and trailing edges, and the 1/4 chord line of the wing.
    Inputs: 
    - chord_root: Root chord of the wing [m]
    - chord_tip: Tip chord of the wing [m]
    - wing_span: Wing span [m]
    - sweep: Sweep angle of the wing [rad]
    Outputs:
    - wing_geometry: Dictionary with the coordinates of the leading and trailing edges, and the 1/4 chord line.
    """
    # Coordinates of the quarter chord line 
    x_quarter_chord = np.array([chord_root/4, np.tan(sweep)*(wing_span/2 + chord_root/4)])
    y_quarter_chord = np.array([0, wing_span/2])

    # Coordinates of the leading edge
    x_leading_edge = np.array([0, x_quarter_chord[1]-chord_tip/4])
    y_leading_edge = np.array([0, wing_span/2])

    # Coordinates of the trailing edge
    x_trailing_edge = np.array([chord_root, chord_tip + x_leading_edge[1]])
    y_trailing_edge = np.array([0, wing_span/2])

    # Concatenate the coordinates to form the wing geometry
    x_leading_edge = np.concatenate((x_leading_edge[::-1], x_leading_edge))
    y_leading_edge = np.concatenate((y_leading_edge - wing_span/2, y_leading_edge))

    x_trailing_edge = np.concatenate((x_trailing_edge[::-1], x_trailing_edge))
    y_trailing_edge = np.concatenate((y_trailing_edge - wing_span/2, y_trailing_edge))

    x_quarter_chord = np.concatenate((x_quarter_chord[::-1], x_quarter_chord))
    y_quarter_chord = np.concatenate((y_quarter_chord - wing_span/2, y_quarter_chord))

    # Store the wing geometry in a dictionary
    wing_geometry = {
        'leading_edge': (x_leading_edge, y_leading_edge), 
        'trailing_edge': (x_trailing_edge, y_trailing_edge), 
        'quarter_chord': (x_quarter_chord, y_quarter_chord)}
    
    return wing_geometry

# Discretize the wing in panels
def interpolate_wing_points(wing_geometry, n, m):
    """
    Interpolates points between the leading and trailing edges, and along the chord line of the wing.
    Inputs:
    - wing_geometry: Dictionary with the coordinates of the leading and trailing edges, and the 1/4 chord line.
    - n: Number of points to interpolate along the span of the wing
    - m: Number of points to interpolate along the chord of the wing
    Outputs:
    - wing_vertical_lines: Coordinates of the points between the leading and trailing edges
    - wing_horizontal_lines: Coordinates of the points along the chord line
    """
    # Interpolate points between the leading and trailing edges
    leading_edge, trailing_edge, quarter_chord = (wing_geometry[key] for key in ['leading_edge', 'trailing_edge', 'quarter_chord'])

    y_interp_leading = np.linspace(leading_edge[1][0], leading_edge[1][-1], n)
    y_interp_trailing = np.linspace(trailing_edge[1][0], trailing_edge[1][-1], n)

    x_interp_leading = np.interp(y_interp_leading, leading_edge[1], leading_edge[0])
    x_interp_trailing = np.interp(y_interp_trailing, trailing_edge[1], trailing_edge[0])

    wing_vertical_lines = np.column_stack((y_interp_leading, y_interp_trailing, x_interp_leading, x_interp_trailing))

    # Interpolate points along the chord line
    chord_interp = np.linspace(quarter_chord[1][0], quarter_chord[1][-1], m)
    wing_horizontal_lines = []

    for y_chord in chord_interp:
        x_interp_chord = x_interp_leading * (1 - (y_chord - quarter_chord[1][0]) / (quarter_chord[1][-1] - quarter_chord[1][0])) + \
                         x_interp_trailing * ((y_chord - quarter_chord[1][0]) / (quarter_chord[1][-1] - quarter_chord[1][0]))
        wing_horizontal_lines.append(x_interp_chord)

    wing_horizontal_lines = np.array(wing_horizontal_lines)

    return wing_vertical_lines, wing_horizontal_lines

# Calculate the length of a panel
def calculate_panel_length(panel):
    """
    Calculates the length between the two points of a panel.
    Inputs:
    - panel: List with the coordinates of the four points of the panel
    Outputs:
    - length: Euclidean length between the two points
    """

    # Extract the coordinates of the two points
    coord1 = panel[0]
    coord2 = panel[3]

    # Calculate the length between the two points
    length = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    
    return length

# Append the length of each panel to a list
def calculate_panel_lengths(panels):
    """
    Calculates the length of each panel in a list of panels.
    Inputs:
    - panels: List with the coordinates of the four points of each panel
    Salida:
    - panel_lengths: List with the lengths of each panel
    """
    panel_lengths = []
    for panel in panels:
        length = calculate_panel_length(panel)
        panel_lengths.append(length)
    return panel_lengths

# Generate the wing panels
def generate_wing_panels(wing_vertical_lines, wing_horizontal_lines):
    """
    Generates the panels of the wing using the coordinates of the points between the leading and trailing edges, and along the chord line.
    [x1, y1], [x2, y2], [x3, y3], [x4, y4]. Also, calculates the quarter chord lines and the midpoints at 3/4 of the chord.
    Inputs:
    - wing_vertical_lines: Coordinates of the points between the leading and trailing edges
    - wing_horizontal_lines: Coordinates of the points along the chord line
    Outputs:
    - panel_data: List with the information of each panel
    - panel_lengths: List with the lengths of each panel
    """
    panels = []
    quarter_chord_lines = []
    three_quarter_chord_midpoints = []
    panel_data = []

    # pre-calculate the increments for the quarter and three quarter chord lines
    quarter_increment = 1/4
    three_quarter_increment = 3/4

    for i in range(len(wing_vertical_lines) - 1):
        for j in range(len(wing_horizontal_lines) - 1):
            panel = [
                [wing_vertical_lines[i, 1], wing_horizontal_lines[j, i]],  # Coordinate 1
                [wing_vertical_lines[i, 0], wing_horizontal_lines[j + 1, i]],  # Coordinate 2
                [wing_vertical_lines[i + 1, 0], wing_horizontal_lines[j + 1, i + 1]],  # Coordinate 3
                [wing_vertical_lines[i + 1, 1], wing_horizontal_lines[j, i + 1]],  # Coordinate 4
            ]
            panels.append(panel)

            # Calculate the quarter chord lines
            q_x1 = wing_vertical_lines[i, 1]
            q_y1 = wing_horizontal_lines[j, i] + quarter_increment * (wing_horizontal_lines[j + 1, i] - wing_horizontal_lines[j, i])
            q_x2 = wing_vertical_lines[i + 1, 1]
            q_y2 = wing_horizontal_lines[j, i + 1] + quarter_increment * (wing_horizontal_lines[j + 1, i + 1] - wing_horizontal_lines[j, i + 1])
            quarter_chord_lines.append([[q_x1, q_y1], [q_x2, q_y2]])

            # Calculate the midpoints at 3/4 of the chord
            three_q_x1 = wing_vertical_lines[i, 1]
            three_q_y1 = wing_horizontal_lines[j, i] + three_quarter_increment * (wing_horizontal_lines[j + 1, i] - wing_horizontal_lines[j, i])
            three_q_x2 = wing_vertical_lines[i + 1, 1]
            three_q_y2 = wing_horizontal_lines[j, i + 1] + three_quarter_increment * (wing_horizontal_lines[j + 1, i + 1] - wing_horizontal_lines[j, i + 1])
            three_quarter_chord_midpoints.append([(three_q_x1 + three_q_x2) / 2, (three_q_y1 + three_q_y2) / 2])

    for idx, panel in enumerate(panels):
        panel_info = (idx+1, panel, quarter_chord_lines[idx], three_quarter_chord_midpoints[idx])
        panel_data.append(panel_info)
    
    # Calculate the lengths of the panels
    panel_lengths = calculate_panel_lengths(panels)

    return panel_data, panel_lengths

# Function to generate the NACA airfoil
def naca_airfoil(NACA, chord, alpha_rad, n=100):
    """
    Generates the coordinates of a NACA 4-digit airfoil.
    Inputs:
    - NACA: 4-digit NACA code
    - chord: Chord length of the airfoil
    - alpha_rad: Angle of attack of the airfoil [rad]
    - n: Number of points to generate along the chord
    Outputs:
    - x: x-coordinates of the airfoil
    - z_extr: z-coordinates of the extrados
    - z_intr: z-coordinates of the intrados
    - z_c: z-coordinates of the mean camber line
    - e_c: Maximum thickness of the airfoil
    """

    f_c = int(NACA[0])/(100.0)   # Punto de máxima curvatura
    xf_c = int(NACA[1])/(10.0)   # Punto de espesor máximo
    e_c = int(NACA[2:])/(100.0)  # Espesor máximo

    x_c = np.linspace(0, chord, n)

    # Equation of the mean camber line
    z_c = np.where(x_c <= xf_c, (f_c*x_c/xf_c**2)*(2*xf_c - x_c), (f_c*(1 - x_c)/(1 - xf_c)**2)*(1 - 2*xf_c + x_c))

    # Equation of the thickness distribution
    z_e = 5*e_c*(0.2969*np.sqrt(x_c) - 0.1260*x_c - 0.3516*x_c**2 + 0.2843*x_c**3 - 0.1036*x_c**4)

    # Compute the coordinates of the airfoil
    x = chord * x_c
    z_extr = (z_c+z_e)*chord*np.cos(alpha_rad)-x*np.sin(alpha_rad)
    z_intr = (z_c-z_e)*chord*np.cos(alpha_rad)-x*np.sin(alpha_rad)
    z_c    = z_c*np.cos(alpha_rad)-x*np.sin(alpha_rad)

    return x, z_extr, z_intr, z_c, e_c

def naca_airfoil_dzdx(NACA, select, alpha, chord, n=100):
    """
    Calculates the derivative of the NACA 4-digit airfoil.
    Inputs:
    - NACA: 4-digit NACA code
    - select: y-coordinate to calculate the derivative
    - alpha: Angle of attack of the airfoil [rad]
    - chord: Chord length of the airfoil
    - n: Number of points to generate along the chord
    Outputs:
    - dz_dx: Derivative of the airfoil at the selected y-coordinate
    """
    x = select

    f_c = int(NACA[0])/(100.0)   # Punto de máxima curvatura
    xf_c = int(NACA[1])/(10.0)   # Punto de espesor máximo

    # Equation of the mean camber line
    dz_dx = np.where(x <= xf_c, (f_c*x/xf_c**2)*(2*xf_c - x), (f_c*(1 - x)/(1 - xf_c)**2)*(1 - 2*xf_c + x))
    dz_dx = (dz_dx*np.cos(alpha) - x * np.sin(alpha))

    return dz_dx                               

# Calculate the curvature in each control point
def curvature(NACA_root, NACA_tip, wing_horizontal_points, alpha, panel_data, chord_root, chord_tip, n, m):
    """
    Calculates the curvature in each control point.
    Inputs:
    - NACA_root: 4-digit NACA code of the root airfoil
    - NACA_tip: 4-digit NACA code of the tip airfoil
    - wing_horizontal_points: Coordinates of the points along the chord line
    - alpha: Angle of attack of the wing [rad]
    - panel_data: List with the information of each panel
    - chord_root: Chord length of the root airfoil
    - chord_tip: Chord length of the tip airfoil
    - n: Number of points to interpolate along the span of the wing
    - m: Number of points to interpolate along the chord of the wing
    Outputs:
    - dz_dx_simetric: List with the curvature in each control point
    """

    # Almacenar las coordenadas de los primeros puntos de control y rellenar el resto con esos mismos puntos
    y_leading_edge_ = wing_horizontal_points[0]
    y_leading_edge = np.tile(y_leading_edge_, len(panel_data))
    
    # Calculate the curvature in each control point
    tip_panels = panel_data[:][0:m]
    root_panels = panel_data[:][(int(m*n/2)):(int(m*n/2)+m)]
    y_coord_tip_panels = [coord[-1][1] for coord in tip_panels] - y_leading_edge[0]
    y_coord_root_panels = [coord[-1][1] for coord in root_panels]
    
    dz_dx_tip = []
    dz_dx_root = []

    for select in y_coord_tip_panels:
        dz_dx_tip = np.append(dz_dx_tip, naca_airfoil_dzdx(NACA_tip, select, alpha, chord_tip, n=100))

    for select in y_coord_root_panels: 
        dz_dx_root = np.append(dz_dx_root, naca_airfoil_dzdx(NACA_root, select, alpha, chord_root, n=100))

    # Interpolating the curvature between the root and tip airfoils
    dz_dx = np.zeros((len(dz_dx_tip), int(n/2)))
    for i in range(len(dz_dx_tip)):
        dz_dx[i] = np.linspace(dz_dx_tip[i], dz_dx_root[i], int(n/2))

    dz_dx_simetric = np.hstack((dz_dx, dz_dx[:, ::-1]))

    # Convert the matrix into a vector (by stacking its columns).
    dz_dx_simetric = dz_dx_simetric.ravel('F').tolist()

    return dz_dx_simetric


# Calculate the local angle of attack in each control point
def alphas(dz_c, panel_data, alpha):
    """
    Calcula el ángulo de ataque local en cada punto de control.
    Entradas:
    - dz_c: Lista con la curvatura en cada punto de control
    - panel_data: Lista con la información de cada panel
    - alpha: Ángulo de ataque del ala
    Salida:
    - alphas: Lista con el ángulo de ataque local en cada punto de control
    """
    # List to store the local angle of attack in each control point
    alphas = []

    # Calculate the local angle of attack in each control point
    for i, panel in enumerate(panel_data):
        # Calcular el ángulo de ataque local
        alpha_i = (alpha + dz_c[i])
        alphas.append(alpha_i)

    return np.array(alphas)

# Calculate the aerodynamic influence coefficient matrix
def calculate_P_ij(panel_data):
    """
    Calculates the coefficient matrix P_ij for the VLM method.
    Inputs:
    - panel_data: List with the information of each panel
    Outputs:
    - P_ij: Coefficient matrix P_ij
    """
    n_controls = len(panel_data)
    P_ij = np.zeros((n_controls, n_controls))

    for i, panel_i in enumerate(panel_data):
        xi, yi = panel_i[3]  # Control point for panel i
        for j, panel_j in enumerate(panel_data):
            xj, yj = panel_j[2][0]  # Node 1 of panel j
            xjf, yjf = panel_j[2][1]  # Node 2 of panel j

            a, b = yi - yj, xi - xj
            c, d = yi - yjf, xi - xjf
            e, f = (a**2 + b**2)**0.5, (c**2 + d**2)**0.5
            g, h = yjf - yj, xjf - xj
            div = a * d - c * b

            v1 = - (1 + a / e ) / (4 * np.pi * b) if b != 0 else 0
            v2 = + (1 + c / f ) / (4 * np.pi * d) if d != 0 else 0
            #v1 = -(a + e) / (b * e) / (4 * np.pi) if b != 0 else 0
            #v2 = (c + f) / (d * f) / (4 * np.pi) if d != 0 else 0
            v3 = (1 / div) * (((g * a + h * b) / e) - ((g * c + h * d) / f)) / (4 * np.pi) if div != 0 else 0

            P_ij[i, j] = (v1 + v2 + v3)
            
    return P_ij

def wing(panel_data, u, dz_c, alphas, rho, sweep, panel_lengths, n, m, wing_span):
    # Calculate the induced velocities at each control point
    w_i = np.zeros(len(panel_data))
    for i in range(len(panel_data)):
        w_i[i] = u * ((dz_c[i]) - (alphas))

    # Calculate P_ij (Aerodynamic Influence Coefficient matrix)
    P_ij = calculate_P_ij(panel_data)

    # Calculate gamma using matrix inverse and multiplication
    gammas = np.linalg.solve(P_ij, w_i)

    # Calculate the lift for each circulation value
    lift = rho * u * gammas * np.cos(sweep) * panel_lengths
    lift_matrix = np.tile(lift.reshape(n, m), 1)
    lift_sum = np.sum(lift_matrix, axis=1)
    lift_total = np.sum(lift_sum)

    print("Total Lift:", lift_total)

    CL = lift_total / (0.5 * rho * u ** 2 * wing_span)
    print("CL:", CL)
    
    return w_i, P_ij, gammas, lift_total, lift_sum, CL

def plot_wing(wing_geometry, wing_span):
    """
    Plot the wing geometry in the x-y plane.
    """
    
    x_leading_edge, y_leading_edge = wing_geometry['leading_edge']
    x_trailing_edge, y_trailing_edge = wing_geometry['trailing_edge']
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_leading_edge, x_leading_edge, 'b-', label='Borde de ataque')
    plt.plot(y_trailing_edge, x_trailing_edge, 'r-', label='Borde de salida')

    # Draw the wing airfoil lines
    plt.plot([0, 0], [x_leading_edge[1], x_trailing_edge[1]], 'k-.', label="Encastre")
    plt.plot([wing_span/2, wing_span/2], [x_leading_edge[0], x_trailing_edge[0]], 'k-', label="Punta del ala")
    plt.plot([-wing_span/2, -wing_span/2], [x_leading_edge[0], x_trailing_edge[0]], 'k-')

    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Ala en planta')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_wing_discretization(panel_data, wing_geometry):
    """
    Plot the wing discretization in panels.
    """
    panels = [item[1] for item in panel_data]
    nodes = [item[2] for item in panel_data]
    controls = [item[3] for item in panel_data]
    
    x_leading_edge, y_leading_edge = wing_geometry['leading_edge']
    x_trailing_edge, y_trailing_edge = wing_geometry['trailing_edge']
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each panel
    for panel in panels:
        polygon = Polygon(panel, closed=True, edgecolor='k', facecolor='white', alpha=1)
        ax.add_patch(polygon)

    # Plot the midpoints at 1/4 and 3/4 of the chord
    for midpoint in controls:
        plt.plot(midpoint[0], midpoint[1], 'ro', markersize='4', zorder=20)
    plt.plot(controls[1][0], controls[1][1], 'ro', markersize=4, label='Puntos de control')

    # Plot the lines at 1/4 of the chord of each panel
    for line in nodes:
        plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='blue', alpha=0.6)
    plt.plot([0,0],[1,1], color='blue', label='Líneas 1/4', alpha=0.6)

    # Plot the wing
    plt.plot(y_leading_edge, x_leading_edge, 'b-', label='Borde de ataque')
    plt.plot(y_trailing_edge, x_trailing_edge, 'r-', label='Borde de salida')

    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    plt.title('Discretización del ala')
    plt.grid(True)
    plt.show()

def plot_wing_heatmap(panel_data, gammas, title='Wing Heatmap', legend='Gamma Value'):
    """
    Plot a heatmap of gamma values on the wing planform.
    """
    plt.figure(figsize=(12, 6))
    
    # Extract coordinates of the control points
    x_controls = [control[3][0] for control in panel_data]
    y_controls = [control[3][1] for control in panel_data]
    
    # Plot the heatmap
    scatter_plot = plt.scatter(x_controls, y_controls, c=gammas, cmap='coolwarm', s=80)
    
    # Configure the plot
    cbar = plt.colorbar(scatter_plot, label=legend, shrink=0.5)
    plt.xlabel('x Coordinate [m]')
    plt.ylabel('y Coordinate [m]')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


def plot_wing_lift(panel_data, lift):
    """
    Plot the lift distribution on the wing.
    """
    plt.figure(figsize=(12, 6))
    
    x_controls = [control[3][0] for control in panel_data]
    x_controls = np.array(x_controls)

    x_controls_unique = []
    for x in x_controls:
        if x not in x_controls_unique:
            x_controls_unique.append(x)

    mid_index = int(len(lift)/2)
    lift[mid_index:] = lift[mid_index:][::1]

    # Graficar la distribución de sustentación
    plt.plot(x_controls_unique, lift, 'o-')
    
    # Configuración del gráfico
    plt.xlabel('x [m]')
    plt.ylabel('Lift [N]')
    plt.title('Lift Distribution on the Wing')
    plt.grid(True)
    plt.show()
