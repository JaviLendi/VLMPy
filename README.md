# VLMPy: Vortex Lattice Method App

<p align="center">
    <img src="docs/img/icon.svg" alt="VLMPy Logo" width="120"/>
</p>

**VLMPy** is a modern, open-source Python toolkit and application for aerodynamic analysis of wings using the Vortex Lattice Method (VLM). Designed for engineers, students, and enthusiasts, VLMPy makes it easy to model, visualize, and analyze lift and drag for a wide range of wing configurations in subsonic, inviscid flow.

**Current release:** [v1.1.0](https://github.com/JaviLendi/VLMPy/releases/tag/v1.1.0), featuring a redesigned interface.

**Academic scope:** The TFG (Final Degree Project/thesis) covers VLMPy up to version 1.0.1. Version 1.0.2 and later releases are post-thesis development and are outside the academic project's scope.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Issues](https://img.shields.io/github/issues/JaviLendi/VLMPy.svg)](https://github.com/JaviLendi/VLMPy/issues)
[![Stars](https://img.shields.io/github/stars/JaviLendi/VLMPy.svg)](https://github.com/JaviLendi/VLMPy/stargazers)
[![Forks](https://img.shields.io/github/forks/JaviLendi/VLMPy.svg)](https://github.com/JaviLendi/VLMPy/network/members)
[![Last Commit](https://img.shields.io/github/last-commit/JaviLendi/VLMPy.svg)](https://github.com/JaviLendi/VLMPy/commits/main)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/JaviLendi/VLMPy.svg?include_prereleases)](https://github.com/JaviLendi/VLMPy/releases)
[![Code Size](https://img.shields.io/github/languages/code-size/JaviLendi/VLMPy.svg)](https://github.com/JaviLendi/VLMPy)

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Features](#features)
  - [Screenshots](#screenshots)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Option 1: Windows Executable (Windows 10/11)](#option-1-windows-executable-windows-1011)
  - [Option 2: Install from Source](#option-2-install-from-source)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Submitting Issues](#submitting-issues)
- [License](#license)
- [Author](#author)
- [Acknowledgements](#acknowledgements)

---

## Introduction

**VLMPy** is an open-source Python tool for aerodynamic analysis of wings using the Vortex Lattice Method (VLM). It is designed for aerospace engineers, students, and aviation enthusiasts to model lift and drag in subsonic, inviscid flow conditions. This project began as a Final Year Project in Aerospace Engineering.

---

## Features

- **Accurate VLM Implementation**: Compute aerodynamic coefficients ($C_L$, $C_D$, moments) for wing configurations.
- **Interactive Visualization**: 2D/3D plots of lift distribution, pressure coefficients, and geometry using Plotly and Matplotlib.
- **Customizable Geometry**: Define wing parameters like sweep, dihedral, taper, and control surfaces.
- **Fuselage Geometry**: Define elliptical stations in the app or load previously imported STL meshes. STL import through the app is temporarily unavailable. Fuselages are visual geometry only and do not affect VLM forces or moments.
- **Flexible Flight Conditions**: Analyze multiple angles of attack, sideslip angles, and airspeeds.
- **NACA Airfoil Support**: Generate and analyze NACA airfoil profiles.
- **Modular Architecture**: Extensible Python codebase for custom analyses and integration with NumPy, Matplotlib, and more.
- **User-Friendly Interface**: Intuitive Python API, ready-to-use executables, and Jupyter notebook examples.
- **Comprehensive Documentation**: Step-by-step guides and practical examples.
- **Performance Instrumentation**: Cached VLM stages, vectorized influence assembly, and optimized Plotly rendering. See [the performance validation guide](docs/README-performance.md).

---

### Screenshots

<div align="center">

<table align="center">
   <tr>
      <td><img src="docs/screenshots/APP/bienvenida.png" alt="VLMPy home page on desktop"/></td>
   </tr>
   <tr>
      <td align="center">Home (desktop)</td>
   </tr>
   <tr>
      <td><img src="docs/screenshots/APP/curvatura.png" alt="NACA airfoil definition and live preview" width="100%"/></td>
   </tr>
   <tr>
      <td align="center">NACA Airfoil Definition and Live Preview</td>
   </tr>
   <tr>
      <td><img src="docs/screenshots/APP/discretizacion.png" alt="Wing Definition and 2D Planform Preview" width="100%"/></td>
   </tr>
   <tr>
      <td align="center">Wing Definition and 2D Planform Preview</td>
   </tr>
   <tr>
      <td><img src="docs/screenshots/APP/discretizacion2.png" alt="Wing Definition and 3D Planform Preview" width="100%"/></td>
   </tr>
   <tr>
      <td align="center">Wing Definition and 3D Planform Preview</td>
   </tr>

</table>

<details>
<summary>View all screenshots</summary>

- **2D Panel-Method Pressure Coefficient (NACA 3215)**  
   ![2D panel-method pressure coefficient for NACA 3215](docs/screenshots/APP/velocidad.png)

- **2D Panel-Method results (NACA 3215)**  
   ![2D panel-method results for NACA 3215](docs/screenshots/APP/airfoilresults.png)

- **Plane or Wing Results Panel Dashboard (CL Plot)**  
   ![Plane or Wing Results Panel Dashboard (CL Plot)](docs/screenshots/APP/results1.png)

- **Plane or Wing Results Panel Dashboard (3D Plot)**  
   ![Plane or Wing Results Panel Dashboard (3D Plot)](docs/screenshots/APP/results2.png)

- **3D Wing Discretization (A320)**  
   ![3D Discretization](docs/screenshots/A320/Wing_Discretization3d_htpyvtp1.svg)

- **Vorticity Distribution (Flat Plate with Aileron)**  
   ![Gammas](docs/screenshots/Placa%20plana%20con%20aleron/Gammas.svg)

</details>
</div>

## Installation

### Prerequisites

- **Python 3.8+** (for source installation)
- **Web Browser** (Chrome 90+, Firefox 85+, or equivalent)
- **Git** (optional, for cloning the repository)

### Option 1: Windows Executable (Windows 10/11)

1. Download `VLMPy-Setup-1.1.0.exe` from the [v1.1.0 release](https://github.com/JaviLendi/VLMPy/releases/tag/v1.1.0).
2. Double-click to launch the application—no additional setup required.

### Option 2: Install from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/JaviLendi/VLMPy.git
   cd VLMPy
   ```

2. Install the application dependencies from the repository root:

   ```bash
   python -m pip install -r requirements.txt
   ```

   *Tip*: Update pip with `python -m pip install --upgrade pip` if needed.
3. Run the app:

   ```bash
   python src/vlm/launcher.py
   ```

4. Access the web interface at [http://localhost:5001](http://localhost:5001).

---

## Usage

- **Executable**: Install VLMPy from the Windows installer linked in the [v1.1.0 release](https://github.com/JaviLendi/VLMPy/releases/tag/v1.1.0) for a standalone experience.
- **Python API**: Core modules for custom analyses are in `src/vlm/lib/`.
- **Jupyter Notebooks**: Explore example notebooks in `docs/notebooks/` for guided tutorials.
- **Web Interface**: Run `python src/vlm/launcher.py` to open the desktop interface.

Example notebooks and documentation are available in `docs/notebooks/` and `docs/`.

### Aircraft reference frame

Aircraft coordinates use body axes: x forward, y right, and z down, in metres. The wing leading-edge root is the geometry origin; tail translations and fuselage placement are expressed in these axes. Positive sweep moves the tip aft and positive dihedral raises it. Angles in the Python API are in radians; the app uses degrees. The relative freestream is `-U * (cos(alpha)*cos(beta), sin(beta), sin(alpha)*cos(beta))`. The prescribed wake remains aligned with negative body x.

Configurations saved before this coordinate change need their x and z positions converted before reuse; imported STL vertices and rotations must also be checked against their source frame. Legacy saved results use different axes and cannot be loaded as current results; rerun the analysis. Panel data retain their internal `(y, x, z)` storage order, while the wing preview API and STL export use `(x, y, z)`.

### Fuselage geometry

In Plane Definition, select **Define stations** and use the station controls to enter each x position, y radius, and z radius in metres. Positions must increase along x, from tail towards nose; the end radii must be zero and interior radii positive. Consecutive elliptical sections are connected linearly.

STL import through the app is temporarily unavailable; previously saved imported fuselages can still be loaded. The import API accepts ASCII and binary STL files up to 5 MiB and 50,000 triangles, not native CAD or OpenVSP files. STL does not encode reliable units or component semantics. Record the source tool, frame, and tessellation settings when known; the source file is not modified. Saved configurations contain the imported mesh and source checksum, not the original file.

Placement uses translation in metres and rotations in degrees about fixed aircraft x, y, then z axes, followed by translation. Previously imported STL geometry also supports an explicit positive scale factor about its source origin, applied before rotation and translation. The x axis is longitudinal, y lateral, and z vertical. Review the source frame and explicitly choose the placement: no units or orientation are inferred. Fuselage meshes are previewed with lifting surfaces but are not aerodynamic panels and do not affect the VLM solution. Wing-body interference and fuselage loads are not modeled.

**Generate STL** includes the selected fuselage with the wing and any configured stabilizers. Components are exported as separate, potentially intersecting surfaces in one STL; they are not boolean-merged. STL does not encode units, so treat exported coordinates as metres. Control-surface deflections are not represented in STL exports.

### Tail geometry and controls

Horizontal and vertical stabilizers accept incremental root-to-tip twist and optional span-limited elevator or rudder deflection. Angles in the app are degrees; the Python configuration uses radians. Positive twist moves the horizontal-tail trailing edge toward positive z and the vertical-tail leading edge toward positive y. Control start and end are fractions of the horizontal tail's half-span or the vertical tail's span; the hinge is a chord fraction from the leading edge. Positive elevator deflection moves the trailing edge toward positive z; positive rudder deflection moves it toward positive y. The control geometry is a small-angle approximation, not a nonlinear hinge or separated-flow model. Static control derivatives use central differences per radian. Symmetric horizontal-tail discretization uses an even number of spanwise panels. Tail control response has software and symmetry checks but no external aerodynamic validation.

The Python component classes `Wing`, `HorizontalTail`, and `VerticalTail` in `src/vlm/lib/geometry.py` and `Fuselage` in `src/vlm/lib/fuselage.py` accept the existing configuration dictionaries. They do not replace the public `VLM` plane-dictionary interface.

---

## Project Structure

```text
VLMPy/
├── src/
│   └── vlm/
│       ├── app.py         # Application entry point
│       ├── lib/           # Core VLM and utility modules
│       ├── static/        # Static assets
│       ├── templates/     # HTML templates for web interface
│       └── data/          # Application data and saved configurations
├── docs/
│   ├── notebooks/         # Example Jupyter notebooks
│   └── screenshots/       # Application and analysis screenshots
├── tests/                 # Automated tests
├── README.md             # Project documentation
└── LICENSE               # License information
```

---

## Submitting Issues

Found a bug or have a suggestion? Please open an issue on GitHub.

1. Search existing issues to avoid duplicates.
2. Provide a clear title and detailed description.
3. Include steps to reproduce, expected behavior, and relevant screenshots or logs if possible.

Your feedback helps improve VLMPy!

---

## License

VLMPy is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

---

## Author

**Javier Lendínez**  
*Aerospace Engineer*

- 💼 GitHub: [@JaviLendi](https://github.com/JaviLendi)
- 🏫 Universidad Alfonso X el Sabio

---

## Acknowledgements

- **Supervisor**: Héctor Gómez Cedenilla
- **Institution**: Universidad Alfonso X el Sabio
- **Date**: June 2025

- **Additional Acknowledgements**: Marcos Antonio Rodríguez and Ignacio Maroto Villar, professors at Universidad Alfonso X el Sabio

> ℹ️ For academic details and mathematical derivations, see `docs/TFG.pdf` (coming soon) and source code appendices.

---
