# VLMPy Notebooks

Jupyter notebooks that exercise the public `VLM` API (`src/vlm/lib/vlm.py`,
`src/vlm/lib/geometry.py`) to explain how VLMPy works. Each notebook imports the real
library code directly — nothing here is a mock or a re-implementation.

## Prerequisites

- Python 3.8+ with `numpy`, `plotly`, and `numpy-stl` installed (`src/vlm/lib/geometry.py`
  imports the `stl` package unconditionally, so it is required even though these
  notebooks do not export STL files).
- A Jupyter kernel (VS Code's built-in notebook support or `jupyter lab`/`notebook`).

Every notebook's first code cell searches upward from the current working directory for
`src/vlm` and adds it to `sys.path`, so the notebooks can be launched from any directory.

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
   velocity field. It needs only `numpy`, `matplotlib` and `plotly`.

## Scope and limitations

These notebooks stay within the assumptions documented in `docs/README-agent.md`: steady,
incompressible, inviscid, irrotational flow over thin lifting surfaces at low Mach number.
They do not model viscous drag, stall, separation, or compressibility, and they do not
exercise unsteady VLM, aeroelastic, or flight-dynamics modules because those are roadmap
items in `docs/README-plan.md`, not implemented code.

The additional computational cells in this revision were executed individually with
Python 3.13.15. The complete notebooks were not rerun from a clean kernel. Run every cell
(Run All) and confirm the assertions before treating the full notebook outputs as verified.

Notebook 05 is stored without outputs. Its 12 code cells were executed in order in a plain Python process (plot calls disabled) and all assertions passed; it was not run in a Jupyter kernel.
