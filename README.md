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

- [VLMPy: Vortex Lattice Method App](#vlmpy-vortex-lattice-method-app)
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
- **Flexible Flight Conditions**: Analyze multiple angles of attack, sideslip angles, and airspeeds.
- **NACA Airfoil Support**: Generate and analyze NACA airfoil profiles.
- **Modular Architecture**: Extensible Python codebase for custom analyses and integration with NumPy, Matplotlib, and more.
- **User-Friendly Interface**: Intuitive Python API, ready-to-use executables, and Jupyter notebook examples.
- **Comprehensive Documentation**: Step-by-step guides and practical examples.

---
### Screenshots

<div align="center">

<!-- Gallery container -->
<div style="max-width: 600px; margin: 0 auto;">

<!-- Image display -->
<img id="gallery-image" src="docs/screenshots/APP/bienvenida.png" alt="VLMPy Screenshot 1" style="width:100%; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: 0.3s;"/>

<!-- Caption -->
<p id="gallery-caption" style="font-size: 1rem; color: #555; margin: 0.5em 0 1em 0;">
Bienvenida
</p>

<!-- Navigation buttons -->
<button onclick="prevImage()" style="padding: 0.5em 1.2em; margin-right: 1em; border: none; background: #eee; border-radius: 5px; cursor: pointer;">&#8592; Prev</button>
<button onclick="nextImage()" style="padding: 0.5em 1.2em; border: none; background: #eee; border-radius: 5px; cursor: pointer;">Next &#8594;</button>

</div>

</div>

<!-- Gallery script (works on GitHub Pages, some Markdown renderers, and Jupyter) -->
<script>
const images = [
   { src: "docs/screenshots/APP/bienvenida.png", caption: "Bienvenida" },
   { src: "docs/screenshots/APP/menu.png", caption: "Menú principal" },
   { src: "docs/screenshots/APP/curvatura.png", caption: "Curvatura del ala" },
   { src: "docs/screenshots/A320/Wing_Discretization3d_htpyvtp1.svg", caption: "Discretización 3D del ala (A320)" },
   { src: "docs/screenshots/Placa%20plana%20con%20aleron/Gammas.svg", caption: "Distribución de vorticidad (Placa plana con alerón)" }
];
let idx = 0;
function showImage(i) {
   idx = (i + images.length) % images.length;
   document.getElementById('gallery-image').src = images[idx].src;
   document.getElementById('gallery-image').alt = images[idx].caption;
   document.getElementById('gallery-caption').innerText = images[idx].caption;
}
function prevImage() { showImage(idx - 1); }
function nextImage() { showImage(idx + 1); }
</script>

<details>
<summary>Ver todas las capturas</summary>

|   |   |
|---|---|
| ![Bienvenida](docs/screenshots/APP/bienvenida.png) | ![Menú principal](docs/screenshots/APP/menu.png) |
| ![Curvatura](docs/screenshots/APP/curvatura.png) | ![Discretización 3D](docs/screenshots/A320/Wing_Discretization3d_htpyvtp1.svg) |
| ![Gammas](docs/screenshots/Placa%20plana%20con%20aleron/Gammas.svg) |   |

</details>

## Installation

### Prerequisites

- **Python 3.8+** (for source installation)
- **Web Browser** (Chrome 90+, Firefox 85+, or equivalent)
- **Git** (optional, for cloning the repository)
- 
### Option 1: Windows Executable (Windows 10/11)

1. Download `app.exe` from the [Releases page](https://github.com/JaviLendi/VLMPy/releases).
2. Double-click to launch the application—no additional setup required.

### Option 2: Install from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/JaviLendi/VLMPy.git
   cd VLMPy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Tip*: Update pip with `pip install --upgrade pip` if needed.
3. Run the app:
   ```bash
   python app.py
   ```
4. Access the web interface at [http://localhost:5000](http://localhost:5000).

---
## Usage

- **Executable**: Launch `app.exe` (Windows) for a standalone experience.
- **Python API**: Import VLMPy modules in your scripts for custom analyses (see `src/lib/`).
- **Jupyter Notebooks**: Explore example notebooks in the `notebooks/` directory for guided tutorials.
- **Web Interface**: Run `app.py` to access the interactive web app at [http://localhost:5000](http://localhost:5000).

Example scripts and documentation are available in the `notebooks/` and `docs/` directories.

---

## Project Structure

```
VLMPy/
├── src/                  # Source code
│   ├── lib/              # Core VLM and utility modules
│   │   ├── vlm.py        # Vortex Lattice Method implementation
│   │   ├── naca.py       # NACA airfoil generation utilities
│   │   ├── geometry.py   # Wing geometry creation and manipulation
│   ├── static/           # Static assets (images, data files)
│   ├── templates/        # HTML templates for web interface
│   └── app.py            # Application entry point
├── notebooks/            # Example Jupyter notebooks
├── docs/                 # Documentation and project reports
├── requirements.txt      # Python dependencies
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
_Aerospace Engineer_

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