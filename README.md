# VLMPy: Vortex Lattice Method App

<p align="center">
    <img src="docs/img/icon.svg" alt="VLMPy Logo" width="120"/>
</p>

**VLMPy** is a modern, open-source Python toolkit and application for aerodynamic analysis of wings using the Vortex Lattice Method (VLM). Designed for engineers, students, and enthusiasts, VLMPy makes it easy to model, visualize, and analyze lift and drag for a wide range of wing configurations in subsonic, inviscid flow.

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
  - [External 3D geometry](#external-3d-geometry)
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
- **External Geometry Roadmap**: Plan support for OpenVSP workflows and STL surface-mesh inspection and import, subject to explicit unit, coordinate, component-mapping, and aerodynamic-panelization validation.
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
      <td><img src="docs/screenshots/APP/bienvenida.png" alt="Welcome"/></td>
      <td><img src="docs/screenshots/APP/menu.png" alt="Main Menu"/></td>
   </tr>
   <tr>
      <td align="center">Welcome</td>
      <td align="center">Main Menu</td>
   </tr>
</table>

<p align="center">
   <img src="docs/screenshots/APP/curvatura.png" alt="Wing Curvature" width="48%"/>
   <img src="docs/screenshots/A320/Wing_Discretization3d_htpyvtp1.svg" alt="3D Wing Discretization (A320)" width="48%"/>
</p>

</div>

<details>
<summary>View all screenshots</summary>

- **Welcome**  
   ![Welcome](docs/screenshots/APP/bienvenida.png)

- **Main Menu**  
   ![Main Menu](docs/screenshots/APP/menu.png)

- **Wing Curvature**  
   ![Curvature](docs/screenshots/APP/curvatura.png)

- **3D Wing Discretization (A320)**  
   ![3D Discretization](docs/screenshots/A320/Wing_Discretization3d_htpyvtp1.svg)

- **Vorticity Distribution (Flat Plate with Aileron)**  
   ![Gammas](docs/screenshots/Placa%20plana%20con%20aleron/Gammas.svg)

</details>

## Installation

### Prerequisites

- **Python 3.8+** (for source installation)
- **Web Browser** (Chrome 90+, Firefox 85+, or equivalent)
- **Git** (optional, for cloning the repository)

### Option 1: Windows Executable (Windows 10/11)

1. Download `app.exe` from the [Releases page](https://github.com/JaviLendi/VLMPy/releases).
2. Double-click to launch the application—no additional setup required.

### Option 2: Install from Source

1. Clone the repository:

   ```bash
   git clone https://github.com/JaviLendi/VLMPy.git
   cd VLMPy
   ```

2. Install the application dependencies:

   ```bash
   python -m pip install numpy Flask flask-compress plotly matplotlib numpy-stl psutil PySide6
   ```

   *Tip*: Update pip with `python -m pip install --upgrade pip` if needed.
3. Run the app:

   ```bash
   python src/vlm/app.py
   ```

4. Access the web interface at [http://localhost:5001](http://localhost:5001).

---

## Usage

- **Executable**: Launch `app.exe` (Windows) for a standalone experience.
- **Python API**: Core modules for custom analyses are in `src/vlm/lib/`.
- **Jupyter Notebooks**: Explore example notebooks in `docs/notebooks/` for guided tutorials.
- **Web Interface**: Run `python src/vlm/app.py` to access the interactive web app at [http://localhost:5001](http://localhost:5001).

Example notebooks and documentation are available in `docs/notebooks/` and `docs/`.

### External 3D geometry

The project roadmap includes importing or inspecting aircraft geometry created in OpenVSP and triangular surface meshes in STL format. OpenVSP is intended to provide a parametric aircraft source, while STL is treated as a surface mesh and does not contain complete aerodynamic metadata. Before external geometry can be analyzed, VLMPy must validate units, coordinate axes, scale, surface normals, connected components, and the mapping from mesh regions to aerodynamic components.

This workflow is planned unless a corresponding importer and validation case are present in the current source tree. See `docs/README-plan.md` and `docs/README-agent.md` for the intended requirements.

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

**Javier Lendínez Castillo**  
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
