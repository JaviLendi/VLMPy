# VLMPy Notebooks

Jupyter notebooks that exercise the public `VLM` API (`src/vlm/lib/vlm.py`,
`src/vlm/lib/geometry.py`) to explain how VLMPy works. They use the actual library;
independent reference calculations are identified separately from the solver.

## Prerequisites

- Python 3.8+ with `numpy`, `plotly`, and `numpy-stl` installed (`src/vlm/lib/geometry.py`
  imports the `stl` package unconditionally, so it is required even though these
  notebooks do not export STL files).
- A Jupyter kernel (VS Code's built-in notebook support or `jupyter lab`/`notebook`).

Every notebook's first code cell searches upward from the current working directory for
`src/vlm` and adds it to `sys.path`, so the notebooks can be launched from the repository
root or a directory inside the repository.

Each notebook introduces its theory and workflow before the examples, includes numerical
diagnostics and visualizations, and closes with verification and validation limits,
references, use cases, possible improvements, and next steps. A theoretical comparison
or an internal consistency check is not experimental validation.

See [VALIDATION-PLAN.md](VALIDATION-PLAN.md) for prioritized findings, remaining
verification work, external-validation requirements, and acceptance evidence.

## Contents

1. **`01_steady_vlm_fundamentals.ipynb`** — builds a high-aspect-ratio flat plate,
   compares the lift-curve slope with the thin-airfoil limit, studies spanwise mesh
   sensitivity, and plots an induced-drag polar.
2. **`02_geometry_and_discretization.ipynb`** — builds a tapered, swept, cambered wing
   with a flap, inspects panel records, visualizes the geometry and mesh, compares lift
   and induced drag across flap deflections, checks the mirrored spanwise response to
   antisymmetric aileron deflections, and contrasts a winglet cant sweep with the TFG.
3. **`03_multi_surface_configuration.ipynb`** — couples a wing with horizontal and
   vertical stabilizers, inspects per-surface forces and distributions, and visualizes
   the force response to horizontal-stabilizer incidence and sideslip. It distinguishes
   multisurface scalar outputs from the separate symmetric wing-only sideslip case.
4. **`04_persistence_and_performance.ipynb`** — follows the verification procedure in
   `docs/README-performance.md`, checks cache behavior and persistence round trips, and
   visualizes the influence matrix and solved circulation as numerical diagnostics. It
   also times one cold run at several mesh sizes and interprets the result as a local
   observation, not a benchmark.
5. **`05_panel_methods_2d.ipynb`** — explains the 2D panel-method module
   `src/vlm/lib/panel2d.py` (Hess-Smith sources plus vortex, and doublets with a wake)
   following Katz & Plotkin, Chapter 11. It verifies the influence kernels against
   numerical integration, the Neumann and Kutta residuals, Kutta-Joukowski against the
   pressure integral, convergence with panel count, and the near-field behavior of the
   velocity field. It also compares the NACA 0012 lift slope with a qualified NASA
   correlation; the observed difference is documented as unresolved. It needs only
   `numpy`, `matplotlib` and `plotly`.
6. **`06_aerodynamic_coefficients.ipynb`** — explains the scalar and vector force
   and moment coefficients, reconstructs panel loads, studies mesh
   convergence, and compares lift and induced drag with an independent lifting-line
   calculation. The comparison is theoretical, not experimental validation.
7. **`07_static_aerodynamic_derivatives.ipynb`** — explains the static derivatives
   of force and moment coefficients, verifies central differences and state
   restoration, and examines reference-point, step-size, spanwise, and chordwise mesh
   sensitivity.

## Scope and limitations

These notebooks stay within the assumptions documented in `docs/README-agent.md`: steady,
incompressible, inviscid, irrotational flow over thin lifting surfaces at low Mach number.
They do not model viscous drag, stall, separation, or compressibility, and they do not
exercise unsteady VLM, aeroelastic, or flight-dynamics modules because those are roadmap
items in `docs/README-plan.md`, not implemented code.
