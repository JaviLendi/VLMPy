#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import plotly.graph_objects as go

def naca_airfoil(NACA, cuerda, alpha_rad, n=100):
    """
    Generates the coordinates of a NACA 4-digit airfoil.
    Parameters:
    NACA (str): A string representing the 4-digit NACA airfoil code.
    cuerda (float): The chord length of the airfoil.
    alpha (float): The angle of attack in radians.
    n (int, optional): The number of points to generate along the chord. Default is 100.
    Returns:
    tuple: A tuple containing the following arrays:
        - x (numpy.ndarray): The x-coordinates of the airfoil.
        - z_extr (numpy.ndarray): The z-coordinates of the upper surface of the airfoil.
        - z_intr (numpy.ndarray): The z-coordinates of the lower surface of the airfoil.
        - z_c (numpy.ndarray): The z-coordinates of the camber line.
        - e_c (float): The maximum thickness of the airfoil.
    """

    f_c = int(NACA[0])/(100.0)   # Punto de máxima curvatura (Max Camber)
    xf_c = int(NACA[1])/(10.0)   # Punto de espesor máximo (Max Camber position)
    e_c = int(NACA[2:])/(100.0)  # Espesor máximo (Thickness)

    x_c = np.linspace(0, 1, n)

    # Ecuación de línea media
    z_c = np.where(x_c <= xf_c, (f_c*x_c/xf_c**2)*(2*xf_c - x_c), (f_c*(1 - x_c)/(1 - xf_c)**2)*(1 - 2*xf_c + x_c))

    # Ley de espesores
    z_e = 5*e_c*(0.2969*np.sqrt(x_c) - 0.1260*x_c - 0.3516*x_c**2 + 0.2843*x_c**3 - 0.1036*x_c**4)

    # Coordenadas completas
    x = cuerda * x_c
    z_extr = (z_c+z_e)*cuerda*np.cos(alpha_rad)-x*np.sin(alpha_rad)
    z_intr = (z_c-z_e)*cuerda*np.cos(alpha_rad)-x*np.sin(alpha_rad)
    z_c    = z_c*np.cos(alpha_rad)-x*np.sin(alpha_rad)

    return x, z_extr, z_intr, z_c, z_e, e_c

def naca_airfoil_dzdx(NACA, select):
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

    m = int(NACA[0])/(100.0)   # Punto de máxima curvatura
    p = int(NACA[1])/(10.0)   # Punto de espesor máximo

    # Equation of the mean camber line
    z_c = np.where(x <= p, (m*x/p**2)*(2*p - x), (m*(1 - x)/(1 - p)**2)*(1 - 2*p + x))
    # z_c = np.where(0 <= x <= p, (m/p**2)*(2*p*x - x**2), (m/(1 - p)**2)*(1 - 2*p + 2*p*x - x**2))

    return z_c    

def plot_naca_airfoil(fig, NACA, cuerda, alpha, n=100):
    alpha = np.radians(alpha)  # Convertir a radianes
    print(f"Generando perfil NACA {NACA} con cuerda {cuerda} m y ángulo de ataque {alpha:.2f} rad")
    # Genera las coordenadas del perfil NACA
    x, z_extr, z_intr, z_c, _, _ = naca_airfoil(NACA, cuerda, alpha, n)

    # Add traces for the airfoil surfaces and camber line
    fig.add_trace(go.Scatter(x=x, y=z_extr, mode='lines', name='Upper Surface'))
    fig.add_trace(go.Scatter(x=x, y=z_intr, mode='lines', name='Lower Surface'))
    fig.add_trace(go.Scatter(x=x, y=z_c, mode='lines', name='Camber Line'))

    # Add angle of attack line if alpha is not zero
    if alpha != 0:
        angle_of_attack_line_x = [0, cuerda]
        angle_of_attack_line_y = [0, np.tan(-alpha)]
        fig.add_trace(go.Scatter(x=angle_of_attack_line_x, y=angle_of_attack_line_y, 
                                 mode='lines', name='Angle of Attack', line=dict(dash='dash', color='red')))
        fig.add_annotation(x=0, y=np.tan(-alpha)/2, text=f'α={np.degrees(alpha):.2f}°', 
                           showarrow=False, font=dict(color='red', size=12))

    # Set layout properties
    fig.update_layout(
        title=f'NACA {NACA} Airfoil' + (f' with Angle of Attack {np.degrees(alpha):.2f}°' if alpha != 0 else ''),
        xaxis_title='Chord (m)',
        yaxis_title='Thickness (m)',
        xaxis=dict(scaleanchor="y", scaleratio=1),  # Equal aspect ratio
        yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True
    )

    # Adjust axis limits with margins
    x_margin = cuerda / 10
    z_margin = cuerda / 10
    valid_z_extr = z_extr[np.isfinite(z_extr)]
    valid_z_intr = z_intr[np.isfinite(z_intr)]
    fig.update_xaxes(range=[min(x) - x_margin, max(x) + x_margin])
    fig.update_yaxes(range=[np.min([np.min(valid_z_extr), np.min(valid_z_intr)]) - z_margin, 
                            np.max([np.max(valid_z_extr), np.max(valid_z_intr)]) + z_margin])
    
def naca_csv(NACA, cuerda, alpha, n=100, filename="naca_airfoil.csv"):
    x, _, _, z_c, z_e, e_c = naca_airfoil(NACA, cuerda, alpha, n)

    f_c = int(NACA[0])/(100.0)   # Punto de máxima curvatura (Max Camber)
    xf_c = int(NACA[1])/(10.0)   # Punto de espesor máximo (Max Camber position)
    e_c = int(NACA[2:])/(100.0)  # Espesor máximo (Thickness)

    x_c = np.linspace(0, 1, n)
    z_extr = z_c + z_e
    z_intr = z_c - z_e

    x_coords = np.concatenate((np.flip(x_c), x_c))
    z_coords = np.concatenate((np.flip(z_extr), (z_intr)))

    # Save to file
    with open(filename, 'w') as file:
        file.write(f"NACA {NACA} Airfoil M={f_c*100:.1f}% P={xf_c*10:.1f}% T={e_c*100:.1f}%\n")
        for x, y in zip(x_coords, z_coords):
            file.write(f"{x: .6f}  {y: .6f}\n")
