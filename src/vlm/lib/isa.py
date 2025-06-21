#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISA functions.
--------------

International Standard Atmosphere (ISA) functions to calculate temperature, pressure, density and altitude
as a function of the geometric altitude above sea level.

The International Standard Atmosphere (ISA) is a static model of the Earth's atmosphere that describes how the
pressure, temperature, density, and viscosity of the Earth's atmosphere change with altitude. The model, based on
a mid-latitude, summer-time average atmospheric conditions, consists of a series of layers with different temperature

Contents:
    - isa_temperature(altitude, isa_deviation=0)
    - isa_pressure(altitude)
    - isa_altitude(pressure)
    - isa_density(altitude)

References:
    [1] US Standard Atmosphere, 1976, NASA.
    [2] MRod5, "PyTurb: A Python Library for Gas Turbine Analysis", 2021.

Author: Javier Lendínez Castillo
Date: March 2024

"""

def isa_temperature(altitude, isa_deviation=0):
    """
    ISA temperature:
    
    Calculates the International Standard Atmosphere temperature as a function of the
    altitude above sea level.
    
    Parameters:
        altitude (float): Geometric altitude [m]
        isa_deviation (float): Standard day base temperature deviation [K]
        
    Returns:
        float: Static temperature at altitude "altitude" [K]
    """
    if altitude <= 11000:
        base_temperature = 288.15 + isa_deviation  # Base temperature at sea level [K]
        temperature_gradient = 0.0065  # Temperature lapse rate [K/m]
        return base_temperature - temperature_gradient * altitude
    else:
        print(f"Layer not implemented yet for altitude {altitude}")
        return None


def isa_pressure(altitude):
    """
    ISA pressure:
    
    Calculates the International Standard Atmosphere pressure as a function of the altitude
    above sea level.

    Parameters:
        altitude (float): Geometric altitude [m]
        
    Returns:
        float: Static pressure at altitude "altitude" [Pa]
    """
    if altitude <= 11000:
        base_pressure = 101325  # Sea level base pressure [Pa], standard day
        temperature_gradient = 0.0065  # Temperature lapse rate [K/m]
        base_temperature = 288.15  # Base temperature at sea level [K], standard day

        # Constants:
        gravity = 9.80665  # Gravitational acceleration [m/s^2]
        gas_constant = 287.053  # Specific gas constant for dry air [J/(kg*K)]

        # Calculate pressure:
        pressure = base_pressure * ((1 - temperature_gradient * altitude / base_temperature) ** (gravity / gas_constant / temperature_gradient))
        
        return pressure
    else:
        print(f"Layer not implemented yet for altitude {altitude}")
        return None


def isa_altitude(pressure):
    """
    ISA Altitude:
    
    Calculates the altitude above sea level, assuming the International Standard Atmosphere pressure model.
    
    Parameters:
        pressure (float): Static pressure at a given altitude [Pa]
        
    Returns:
        float: Altitude over sea level [m]
    """
    base_pressure = 101325  # Sea level base pressure [Pa], standard day
    temperature_gradient = 0.0065  # Temperature lapse rate [K/m]
    base_temperature = 288.15  # Base temperature at sea level [K], standard day

    # Constants:
    gravity = 9.80665  # Gravitational acceleration [m/s^2]
    gas_constant = 287.05  # Specific gas constant for dry air [J/(kg*K)]

    altitude = base_temperature / temperature_gradient * (1 - ((pressure / base_pressure) ** (gas_constant * temperature_gradient / gravity)))

    return altitude

def isa_density(altitude):
    """
    ISA Density:
    
    Calculates the air density as a function of the altitude above sea level.
    
    Parameters:
        altitude (float): Geometric altitude [m]
        
    Returns:
        float: Air density at altitude "altitude" [kg/m^3]
    """
    if altitude <= 11000:
        base_pressure = 101325  # Sea level base pressure [Pa], standard day
        temperature_gradient = 0.0065  # Temperature lapse rate [K/m]
        base_temperature = 288.15  # Base temperature at sea level [K], standard day

        # Constants:
        gravity = 9.80665  # Gravitational acceleration [m/s^2]
        gas_constant = 287.053  # Specific gas constant for dry air [J/(kg*K)]

        # Calculate pressure:
        pressure = base_pressure * ((1 - temperature_gradient * altitude / base_temperature) ** (gravity / gas_constant / temperature_gradient))

        # Calculate density:
        density = pressure / (gas_constant * isa_temperature(altitude))  # [kg/m^3]
        
        return density
    else:
        print(f"Layer not implemented yet for altitude {altitude}")
        return None

def delta(altitude):
    '''
    ISA delta:
    
    Calculates the ratio of the pressure at a given altitude to the sea level pressure.

    Parameters:
        altitude (float): Geometric altitude [m]

    Returns:
        float: Pressure ratio at altitude "altitude" [-]
    '''
    return isa_pressure(altitude) / 101325

def theta(altitude):
    '''
    ISA theta:

    Calculates the ratio of the temperature at a given altitude to the sea level temperature.

    Parameters:
        altitude (float): Geometric altitude [m]

    Returns:
        float: Temperature ratio at altitude "altitude" [-]
    '''
    return isa_temperature(altitude) / 288.15

def G_ISA(G, h):
    '''
    ISA G:

    Calculates the corrected mass flow rate at a given altitude using the pressure ratio.

    Parameters:
        G (float): Mass flow rate at sea level [kg/s]

    Returns:
        float: Corrected mass flow rate at a given altitude [kg/s]
    '''
    return G * theta(h)**0.5 / delta(h)

def theta(altitude):
    '''
    ISA theta:

    Calculates the ratio of the temperature at a given altitude to the sea level temperature.

    Parameters:
        altitude (float): Geometric altitude [m]

    Returns:
        float: Temperature ratio at altitude "altitude" [-]
    '''

    return isa_temperature(altitude) / isa_temperature(0)

def delta(altitude):
    '''
    ISA delta:

    Calculates the ratio of the pressure at a given altitude to the sea level pressure.

    Parameters:
        altitude (float): Geometric altitude [m]

    Returns:
        float: Pressure ratio at altitude "altitude" [-]
    '''

    return isa_pressure(altitude) / isa_pressure(0)