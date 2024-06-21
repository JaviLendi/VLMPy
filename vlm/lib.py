import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import csv
import chardet

# Calcular la geometría del ala
def calculate_wing_geometry(chord_root, chord_tip, wing_span, sweep):
    """
    Calcula las coordenadas de los bordes de ataque y salida del ala, así como la línea 1/4 de la cuerda.
    Entradas: 
    - chord_root: Cuerda de la raíz del ala [m]
    - chord_tip: Cuerda de la punta del ala [m]
    - wing_span: Envergadura del ala [m]
    - sweep: Flecha del ala [rad]
    Salidas:
    - wing_geometry: Diccionario con las coordenadas de los bordes de ataque y salida, y la línea 1/4 de la cuerda.
    """
    # Coordenadas de la línea 1/4 de cuerda
    x_quarter_chord = np.array([chord_root/4, np.tan(sweep)*(wing_span/2 + chord_root/4)])
    y_quarter_chord = np.array([0, wing_span/2])

    # Coordenadas del borde de ataque
    x_leading_edge = np.array([0, x_quarter_chord[1]-chord_tip/4])
    y_leading_edge = np.array([0, wing_span/2])

    # Coordenadas del borde de salida
    x_trailing_edge = np.array([chord_root, chord_tip + x_leading_edge[1]])
    y_trailing_edge = np.array([0, wing_span/2])

    # Coordenadas del semiala izquierda por simetría
    x_leading_edge = np.concatenate((x_leading_edge[::-1], x_leading_edge))
    y_leading_edge = np.concatenate((y_leading_edge - wing_span/2, y_leading_edge))

    x_trailing_edge = np.concatenate((x_trailing_edge[::-1], x_trailing_edge))
    y_trailing_edge = np.concatenate((y_trailing_edge - wing_span/2, y_trailing_edge))

    x_quarter_chord = np.concatenate((x_quarter_chord[::-1], x_quarter_chord))
    y_quarter_chord = np.concatenate((y_quarter_chord - wing_span/2, y_quarter_chord))

    # Almacenar coordenadas del borde de ataque y el borde de salida en una lista de tuplas
    wing_geometry = {
        'leading_edge': (x_leading_edge, y_leading_edge), 
        'trailing_edge': (x_trailing_edge, y_trailing_edge), 
        'quarter_chord': (x_quarter_chord, y_quarter_chord)}
    
    return wing_geometry

# Discretizar el ala en paneles
def interpolate_wing_points(wing_geometry, n, m):
    """
    Interpola los puntos entre los bordes de ataque y salida a lo largo de la envergadura del ala,
    así como a lo largo de la cuerda.
    Entradas:
    - wing_geometry: Diccionario con las coordenadas de los bordes de ataque y salida, y la línea 1/4 de la cuerda.
    - n: Número de puntos a interpolar a lo largo de la envergadura
    - m: Número de puntos a interpolar a lo largo de la cuerda
    Salidas:
    - wing_vertical_lines: Coordenadas de los puntos entre los bordes de ataque y salida
    - wing_horizontal_lines: Coordenadas de los puntos a lo largo de la cuerda
    """
    # Interpolar puntos entre los bordes de ataque y fuga a lo largo de la envergadura del ala
    leading_edge, trailing_edge, quarter_chord = (wing_geometry[key] for key in ['leading_edge', 'trailing_edge', 'quarter_chord'])

    y_interp_leading = np.linspace(leading_edge[1][0], leading_edge[1][-1], n)
    y_interp_trailing = np.linspace(trailing_edge[1][0], trailing_edge[1][-1], n)

    x_interp_leading = np.interp(y_interp_leading, leading_edge[1], leading_edge[0])
    x_interp_trailing = np.interp(y_interp_trailing, trailing_edge[1], trailing_edge[0])

    wing_vertical_lines = np.column_stack((y_interp_leading, y_interp_trailing, x_interp_leading, x_interp_trailing))

    # Interpolar puntos a lo largo de la cuerda
    chord_interp = np.linspace(quarter_chord[1][0], quarter_chord[1][-1], m)
    wing_horizontal_lines = []

    for y_chord in chord_interp:
        x_interp_chord = x_interp_leading * (1 - (y_chord - quarter_chord[1][0]) / (quarter_chord[1][-1] - quarter_chord[1][0])) + \
                         x_interp_trailing * ((y_chord - quarter_chord[1][0]) / (quarter_chord[1][-1] - quarter_chord[1][0]))
        wing_horizontal_lines.append(x_interp_chord)

    wing_horizontal_lines = np.array(wing_horizontal_lines)

    return wing_vertical_lines, wing_horizontal_lines

# Calcula la anchura de cada panel
def calculate_panel_length(panel):
    """
    Calcula la longitud euclidiana entre los dos puntos extremos del panel.
    Entradas: 
    - panel: Coordenadas de los cuatro puntos extremos del panel
    Salida:
    - length: Longitud del panel
    """

    # Obtenemos las coordenadas de los puntos extremos del panel
    coord1 = panel[0]
    coord2 = panel[3]

    # Calculamos la longitud euclidiana entre los dos puntos
    length = np.sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    
    return length

# Anexiona las anchuras de los paneles
def calculate_panel_lengths(panels):
    """
    Calcula la longitud de cada panel en la malla de paneles.
    Entradas:
    - panels: Lista de paneles 
    Salida:
    - panel_lengths: Lista con las longitudes de cada panel
    """
    panel_lengths = []
    for panel in panels:
        length = calculate_panel_length(panel)
        panel_lengths.append(length)
    return panel_lengths

# Generar los paneles del ala
def generate_wing_panels(wing_vertical_lines, wing_horizontal_lines):
    """
    Genera los paneles para la superficie del ala. Cada panel se define por cuatro coordenadas:
    [x1, y1], [x2, y2], [x3, y3], [x4, y4]. Además, se calculan las líneas a 1/4 y 3/4
    de la cuerda de cada panel. 
    Entradas:
    - wing_vertical_lines: Coordenadas de los puntos entre los bordes de ataque y salida
    - wing_horizontal_lines: Coordenadas de los puntos a lo largo de la cuerda
    Salidas:
    - panel_data: Lista con la información de cada panel
    - panel_lengths: Lista con las longitudes de cada panel
    """
    panels = []
    quarter_chord_lines = []
    three_quarter_chord_midpoints = []
    panel_data = []

    # Pre-calcular los incrementos para los puntos a 1/4 y 3/4 de la cuerda
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

            # Calcular las líneas a 1/4
            q_x1 = wing_vertical_lines[i, 1]
            q_y1 = wing_horizontal_lines[j, i] + quarter_increment * (wing_horizontal_lines[j + 1, i] - wing_horizontal_lines[j, i])
            q_x2 = wing_vertical_lines[i + 1, 1]
            q_y2 = wing_horizontal_lines[j, i + 1] + quarter_increment * (wing_horizontal_lines[j + 1, i + 1] - wing_horizontal_lines[j, i + 1])
            quarter_chord_lines.append([[q_x1, q_y1], [q_x2, q_y2]])

            # Calcular los puntos medios a 3/4
            three_q_x1 = wing_vertical_lines[i, 1]
            three_q_y1 = wing_horizontal_lines[j, i] + three_quarter_increment * (wing_horizontal_lines[j + 1, i] - wing_horizontal_lines[j, i])
            three_q_x2 = wing_vertical_lines[i + 1, 1]
            three_q_y2 = wing_horizontal_lines[j, i + 1] + three_quarter_increment * (wing_horizontal_lines[j + 1, i + 1] - wing_horizontal_lines[j, i + 1])
            three_quarter_chord_midpoints.append([(three_q_x1 + three_q_x2) / 2, (three_q_y1 + three_q_y2) / 2])

    for idx, panel in enumerate(panels):
        panel_info = (idx+1, panel, quarter_chord_lines[idx], three_quarter_chord_midpoints[idx])
        panel_data.append(panel_info)
    
    # Calcular las longitudes de los paneles
    panel_lengths = calculate_panel_lengths(panels)

    return panel_data, panel_lengths

# Función para generar las coordenadas un perfil NACA
def naca_airfoil(NACA, cuerda, alpha_rad, n=100):
    """
    Genera las coordenadas de un perfil NACA de 4 dígitos.
    Entradas:
    - NACA: Código NACA de 4 dígitos
    - cuerda: Longitud de la cuerda del perfil
    - alpha: Ángulo de ataque del perfil
    - n: Número de puntos a generar
    Salidas:
    - x: Coordenada x del perfil
    - z_extr: Coordenada z del extrados del perfil
    - z_intr: Coordenada z del intrados del perfil
    - z_c: Coordenada z de la línea media del perfil
    - e_c: Espesor máximo del perfil 
    """

    f_c = int(NACA[0])/(100.0)   # Punto de máxima curvatura
    xf_c = int(NACA[1])/(10.0)   # Punto de espesor máximo
    e_c = int(NACA[2:])/(100.0)  # Espesor máximo

    x_c = np.linspace(0, cuerda, n)

    # Ecuación de línea media
    z_c = np.where(x_c <= xf_c, (f_c*x_c/xf_c**2)*(2*xf_c - x_c), (f_c*(1 - x_c)/(1 - xf_c)**2)*(1 - 2*xf_c + x_c))

    # Ley de espesores
    z_e = 5*e_c*(0.2969*np.sqrt(x_c) - 0.1260*x_c - 0.3516*x_c**2 + 0.2843*x_c**3 - 0.1036*x_c**4)

    # Coordenadas completas
    x = cuerda * x_c
    z_extr = (z_c+z_e)*cuerda*np.cos(alpha_rad)-x*np.sin(alpha_rad)
    z_intr = (z_c-z_e)*cuerda*np.cos(alpha_rad)-x*np.sin(alpha_rad)
    z_c    = z_c*np.cos(alpha_rad)-x*np.sin(alpha_rad)

    return x, z_extr, z_intr, z_c, e_c

def naca_airfoil_dzdx(NACA, select, alpha, chord, n=100):
    """
    Calcula la derivada dz/dx en un punto dado de un perfil NACA de 4 dígitos.

    Entradas:
    - NACA: Código NACA de 4 dígitos.
    - cuerda: Longitud de la cuerda del perfil.
    - select: Coordenada x del punto donde se desea calcular dz/dx.
    - alpha: Ángulo de ataque del perfil.

    Salida:
    - dz_dx: Derivada dz/dx en el punto dado.
    """
    x = select

    f_c = int(NACA[0])/(100.0)   # Punto de máxima curvatura
    xf_c = int(NACA[1])/(10.0)   # Punto de espesor máximo

    # Ecuación de línea media
    dz_dx = np.where(x <= xf_c, (f_c*x/xf_c**2)*(2*xf_c - x), (f_c*(1 - x)/(1 - xf_c)**2)*(1 - 2*xf_c + x))
    dz_dx = (dz_dx*np.cos(alpha) - x * np.sin(alpha))

    return dz_dx                               

# Calcular la curvatura en cada punto de control
def curvature(NACA_root, NACA_tip, wing_horizontal_points, alpha, panel_data, chord_root, chord_tip, n, m):
    """
    Calcula la curvatura en cada punto de control de la malla de paneles.
    Entradas:
    - NACA_root: Código NACA de 4 dígitos del perfil de la raíz
    - NACA_tip: Código NACA de 4 dígitos del perfil de la punta
    - chord_root: Longitud de la cuerda del perfil de la raíz
    - taper_ratio: estrechamiento del ala
    - alpha: Ángulo de ataque del ala
    - panel_data: Lista con la información de cada panel
    Salida:
    - dz_dx: Lista con la curvatura en cada punto de control
    """

    # Almacenar las coordenadas de los primeros puntos de control y rellenar el resto con esos mismos puntos
    y_leading_edge_ = wing_horizontal_points[0]
    y_leading_edge = np.tile(y_leading_edge_, len(panel_data))
    
    # Calcular la curvatura de cada punto de control
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

    # Interpolación de las curvaturas
    dz_dx = np.zeros((len(dz_dx_tip), int(n/2)))
    for i in range(len(dz_dx_tip)):
        dz_dx[i] = np.linspace(dz_dx_tip[i], dz_dx_root[i], int(n/2))

    dz_dx_simetrica = np.hstack((dz_dx, dz_dx[:, ::-1]))

    # Convertir la matriz en un vector (haciendolo por columnas)
    dz_dx_simetrica = dz_dx_simetrica.ravel('F').tolist()

    return dz_dx_simetrica


# Calcular el ángulo de ataque local en cada punto de control
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
    # Lista para almacenar los ángulos de ataque locales
    alphas = []

    # Calcular el ángulo de ataque local en cada punto de control
    for i, panel in enumerate(panel_data):
        # Calcular el ángulo de ataque local
        alpha_i = (alpha + dz_c[i])
        alphas.append(alpha_i)

    return np.array(alphas)

# Calcular la matriz de coeficientes P_ij
def calculate_P_ij(panel_data):
    """
    Calcula la matriz de coeficientes P_ij para el método de VLM.
    Entradas:
    - panel_data: Lista con la información de cada panel
    Salida:
    - P_ij: Matriz de coeficientes P_ij
    """
    n_controls = len(panel_data)
    P_ij = np.zeros((n_controls, n_controls))

    for i, panel_i in enumerate(panel_data):
        xi, yi = panel_i[3]  # Punto de control del panel i
        for j, panel_j in enumerate(panel_data):
            xj, yj = panel_j[2][0]  # Nodo 1 del panel j
            xjf, yjf = panel_j[2][1]  # Nodo 2 del panel j

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
    # Calcular las velocidades inducidas en cada punto de control
    w_i = np.zeros(len(panel_data))
    for i in range(len(panel_data)):
        w_i[i] = u * ((dz_c[i]) - (alphas))

    # Calcular P_ij (Aerodynamic Influence Coefficient matrix)
    P_ij = calculate_P_ij(panel_data)

    # Calcular gamma usando inversa de matriz y multiplicación
    gammas = np.linalg.solve(P_ij, w_i)

    # Calcular la sustentación para cada valor de la circulación
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
    Grafica el ala en planta.
    """
    x_leading_edge, y_leading_edge = wing_geometry['leading_edge']
    x_trailing_edge, y_trailing_edge = wing_geometry['trailing_edge']
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_leading_edge, x_leading_edge, 'b-', label='Borde de ataque')
    plt.plot(y_trailing_edge, x_trailing_edge, 'r-', label='Borde de salida')

    # Dibujar la línea de los perfiles del ala
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
    Grafica la discretización del ala en paneles.
    """
    panels = [item[1] for item in panel_data]
    nodes = [item[2] for item in panel_data]
    controls = [item[3] for item in panel_data]
    
    x_leading_edge, y_leading_edge = wing_geometry['leading_edge']
    x_trailing_edge, y_trailing_edge = wing_geometry['trailing_edge']
    
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar cada panel
    for panel in panels:
        polygon = Polygon(panel, closed=True, edgecolor='k', facecolor='white', alpha=1)
        ax.add_patch(polygon)

    # Graficar los puntos medios de las líneas a 3/4 de la cuerda
    for midpoint in controls:
        plt.plot(midpoint[0], midpoint[1], 'ro', markersize='4', zorder=20)
    plt.plot(controls[1][0], controls[1][1], 'ro', markersize=4, label='Puntos de control')

    # Graficar las líneas a 1/4 de la cuerda de cada panel
    for line in nodes:
        plt.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], color='blue', alpha=0.6)
    plt.plot([0,0],[1,1], color='blue', label='Líneas 1/4', alpha=0.6)

    # Graficar el ala
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

def plot_wing_heatmap(panel_data, gammas, title='Mapa de calor de los valores de Gamma en la planta del ala', legend='Valor de Gamma'):
    """
    Grafica un mapa de calor con los valores de gamma en la planta del ala.
    """
    plt.figure(figsize=(12, 6))
    
    # Extraer coordenadas de los puntos medios de los paneles
    x_controls = [control[3][0] for control in panel_data]
    y_controls = [control[3][1] for control in panel_data]
    
    # Graficar el mapa de calor
    scatter_plot = plt.scatter(x_controls, y_controls, c=gammas, cmap='coolwarm', s=80)
    
    # Configuración del gráfico
    cbar = plt.colorbar(scatter_plot, label=legend, shrink=0.5)  # Ajustar el tamaño de las etiquetas de la barra de color
    plt.xlabel('Coordenada x [m]')
    plt.ylabel('Coordenada y [m]')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()


def plot_wing_lift(panel_data, lift):
    """
    Grafica la curva de sustentación en el ala.
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
    plt.ylabel('Sustentación [N]')
    plt.title('Distribución de sustentación en el ala')
    plt.grid(True)
    plt.show()
