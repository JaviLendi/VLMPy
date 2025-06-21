# VLMPy: Vortex Lattice Method

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

- [VLMPy: Vortex Lattice Method](#vlmpy-vortex-lattice-method)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Steps](#steps)
      - [Windows Executable (Windows 11 an 10)](#windows-executable-windows-11-an-10)
  - [License](#license)
  - [Submitting Issues](#submitting-issues)
  - [Acknowledgements](#acknowledgements)

---

## Introduction

**VLMPy** is an open-source Python tool for aerodynamic analysis of wings using the Vortex Lattice Method (VLM). It is designed for aerospace engineers, students, and aviation enthusiasts to model lift and drag in subsonic, inviscid flow conditions. This project began as a Final Year Project in Aerospace Engineering.

---

## Features

- Compute aerodynamic coefficients ($C_L$, $C_D$, moments) for various wing configurations.
- Interactive 2D/3D visualization with Plotly.
- Customizable wing geometry: sweep, dihedral, taper, control surfaces.
- Analyze multiple flight conditions (angle of attack, sideslip, airspeed).
- NACA airfoil generation and analysis.
- Modular, extensible Python architecture.

---

## Installation

### Prerequisites

- Python 3.8+
- Web browser (Chrome 90+, Firefox 85+, or equivalent)

### Steps

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
     Access at [http://localhost:5000](http://localhost:5000).

#### Windows Executable (Windows 11 an 10)

Download `app.exe` from the [Releases page](https://github.com/JaviLendi/VLMPy/releases) and double-click to launch.

---

## License

VLMPy is distributed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

---

## Submitting Issues

Found a bug or have a suggestion? Please open an issue on GitHub.

1. Search existing issues to avoid duplicates.
2. Provide a clear title and detailed description.
3. Include steps to reproduce, expected behavior, and relevant screenshots or logs if possible.

Your feedback helps improve VLMPy!

---

## Acknowledgements

- **Author:** Javier Lendínez Castillo
- **Supervisor:** Héctor Gómez Cedenilla
- **Date:** June 2025

> ℹ️ For academic details and mathematical derivations, see `docs/TFG.pdf` (coming soon) and source code appendices.

