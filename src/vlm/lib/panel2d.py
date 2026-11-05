"""2D potential-flow panel methods for NACA 4-digit airfoils.

Conventions
-----------
- Frame: x along the chord (LE at x=0), z up, geometry at zero incidence.
  The free stream is rotated instead: V = U (cos(alpha), sin(alpha)).
- Panel nodes run TE(lower) -> LE -> TE(upper), so the outward normal is
  n = (-dz, dx) / L and the tangent t = (dx, dz) / L.
- Cp = 1 - (Vt / U)^2 (incompressible, inviscid, attached flow, no viscous drag).
- Cm is about the quarter chord, normalised by q*c^2 (sign as in the source scripts).

Methods
-------
- "source": Hess-Smith, constant sources + one global vortex, Kutta on the
  trailing-edge panels.
- "doublet": constant doublets with a point-vortex wake at the trailing edge
  (strength mu_N - mu_0), Kutta implied by the wake.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib.path import Path

METHODS = ("source", "doublet")
_EPS = 1e-9


def naca4_panel_nodes(code: str, chord: float, n_panels: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed-contour nodes (TE lower -> LE -> TE upper) with cosine spacing."""
    m = int(code[0]) / 100.0
    p = int(code[1]) / 10.0
    t = int(code[2:]) / 100.0
    half = n_panels // 2 + 1
    xc = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, half)))
    if m == 0.0 or p == 0.0:
        zc = np.zeros_like(xc)
    else:
        zc = np.where(
            xc <= p,
            m * xc / p**2 * (2 * p - xc),
            m * (1 - xc) / (1 - p) ** 2 * (1 - 2 * p + xc),
        )
    zt = 5 * t * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2
                  + 0.2843 * xc**3 - 0.1036 * xc**4)
    x = chord * xc
    zu = chord * (zc + zt)
    zl = chord * (zc - zt)
    return np.concatenate((x[::-1], x[1:])), np.concatenate((zl[::-1], zu[1:]))


class PanelGeometry:
    def __init__(self, x: np.ndarray, z: np.ndarray, chord: float):
        self.x, self.z, self.chord = x, z, chord
        self.n = len(x) - 1
        self.x_mid = 0.5 * (x[:-1] + x[1:])
        self.z_mid = 0.5 * (z[:-1] + z[1:])
        dx, dz = np.diff(x), np.diff(z)
        self.L = np.hypot(dx, dz)
        self.t_x, self.t_z = dx / self.L, dz / self.L
        self.n_x, self.n_z = -dz / self.L, dx / self.L
        theta = np.arctan2(dz, dx)
        self.cos, self.sin = np.cos(theta), np.sin(theta)
        self.i_le = int(np.argmin(self.x_mid))
        self.path = Path(np.column_stack((x, z)))

    def local_coords(self, px: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Panel-local coordinates of points (m,) for all panels -> (m, n)."""
        dx = px[:, None] - self.x[None, :-1]
        dz = pz[:, None] - self.z[None, :-1]
        return self.cos * dx + self.sin * dz, -self.sin * dx + self.cos * dz

    def to_global(self, u: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.cos * u - self.sin * w, self.sin * u + self.cos * w

    def inside(self, px: np.ndarray, pz: np.ndarray) -> np.ndarray:
        return self.path.contains_points(np.column_stack((px, pz)))


def _source_local(xp, zp, L):
    r1 = xp**2 + zp**2
    r2 = (xp - L) ** 2 + zp**2
    u = np.log(r1 / r2) / (4 * np.pi)
    w = (np.arctan2(zp, xp - L) - np.arctan2(zp, xp)) / (2 * np.pi)
    return u, w


def _doublet_local(xp, zp, L):
    r1 = xp**2 + zp**2
    r2 = (xp - L) ** 2 + zp**2
    u = -(zp / r1 - zp / r2) / (2 * np.pi)
    w = (xp / r1 - (xp - L) / r2) / (2 * np.pi)
    return u, w


def _wake_velocity(px, pz, x_te, z_te):
    dx, dz = px - x_te, pz - z_te
    r2 = dx**2 + dz**2
    return -dz / r2 / (2 * np.pi), dx / r2 / (2 * np.pi)


def _safe_zp(zp):
    return np.where(np.abs(zp) < _EPS, np.where(zp < 0, -_EPS, _EPS), zp)


def _surface_influence(geo: PanelGeometry, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Local-frame (u, w) at panel midpoints from unit-strength panels, self term included."""
    xp, zp = geo.local_coords(geo.x_mid, geo.z_mid)
    func = _source_local if kind == "source" else _doublet_local
    u, w = func(xp, zp, geo.L)
    diag = np.arange(geo.n)
    u[diag, diag] = 0.0
    w[diag, diag] = 0.5 if kind == "source" else 2.0 / (np.pi * geo.L)
    return u, w


def _solve_source(geo: PanelGeometry, U: float, alpha: float) -> dict[str, Any]:
    n = geo.n
    us, ws = _surface_influence(geo, "source")
    uv, wv = ws, -us  # vortex is the 90-degree rotation of the source field
    UsG, WsG = geo.to_global(us, ws)
    UvG, WvG = geo.to_global(uv, wv)

    nx, nz = geo.n_x[:, None], geo.n_z[:, None]
    tx, tz = geo.t_x[:, None], geo.t_z[:, None]
    VnS, VnV = UsG * nx + WsG * nz, UvG * nx + WvG * nz
    VtS, VtV = UsG * tx + WsG * tz, UvG * tx + WvG * tz

    ca, sa = np.cos(alpha), np.sin(alpha)
    A = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    A[:n, :n] = VnS
    A[:n, n] = VnV.sum(axis=1)
    b[:n] = -U * (ca * geo.n_x + sa * geo.n_z)

    iL, iU = 0, n - 1
    A[n, :n] = VtS[iL] + VtS[iU]
    A[n, n] = VtV[iL].sum() + VtV[iU].sum()
    b[n] = -U * (ca * (geo.t_x[iL] + geo.t_x[iU]) + sa * (geo.t_z[iL] + geo.t_z[iU]))

    sol = np.linalg.solve(A, b)
    q, gamma = sol[:n], sol[n]
    Vt = U * (ca * geo.t_x + sa * geo.t_z) + VtS @ q + gamma * VtV.sum(axis=1)
    Vn = U * (ca * geo.n_x + sa * geo.n_z) + VnS @ q + gamma * VnV.sum(axis=1)
    return {"A": A, "strength": q, "gamma": gamma, "Vt": Vt,
            "bc_residual": float(np.max(np.abs(Vn))),
            "kutta_residual": float(Vt[iL] + Vt[iU])}


def _solve_doublet(geo: PanelGeometry, U: float, alpha: float) -> dict[str, Any]:
    n = geo.n
    u, w = _surface_influence(geo, "doublet")
    UG, WG = geo.to_global(u, w)
    A = UG * geo.n_x[:, None] + WG * geo.n_z[:, None]

    x_te, z_te = geo.x[0], geo.z[0]
    uw, ww = _wake_velocity(geo.x_mid, geo.z_mid, x_te, z_te)
    vn_wake = uw * geo.n_x + ww * geo.n_z
    A[:, n - 1] += vn_wake
    A[:, 0] -= vn_wake

    ca, sa = np.cos(alpha), np.sin(alpha)
    b = -U * (ca * geo.n_x + sa * geo.n_z)
    # Neumann doublet matrix has rank n-1 (uniform mu is a null vector); min-norm solution
    # leaves Vt and mu_N - mu_0 unchanged.
    mu = np.linalg.lstsq(A, b, rcond=1e-10)[0]

    # Book Eq. 11.37: free stream + induced tangential velocity (panels and wake), plus the
    # self term of Eq. 11.38, -(1/2) dmu/ds here (sign fixed by the kernel convention above).
    s = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(geo.x_mid), np.diff(geo.z_mid)))))
    Vt_ind = (UG * geo.t_x[:, None] + WG * geo.t_z[:, None]) @ mu
    Vt_wake = (mu[-1] - mu[0]) * (uw * geo.t_x + ww * geo.t_z)
    Vt = U * (ca * geo.t_x + sa * geo.t_z) + Vt_ind + Vt_wake - 0.5 * np.gradient(mu, s, edge_order=2)
    return {"A": A, "strength": mu, "gamma": float(mu[-1] - mu[0]), "Vt": Vt,
            "bc_residual": float(np.max(np.abs(A @ mu - b))),
            "kutta_residual": float("nan")}


def velocity_field(geo: PanelGeometry, method: str, sol: dict[str, Any], U: float,
                   alpha: float, px: np.ndarray, pz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total velocity (u, w) at points (m,)."""
    xp, zp = geo.local_coords(px, pz)
    zp = _safe_zp(zp)
    strength = sol["strength"]
    if method == "source":
        us, ws = _source_local(xp, zp, geo.L)
        u_loc = strength * us + sol["gamma"] * ws
        w_loc = strength * ws - sol["gamma"] * us
    else:
        ud, wd = _doublet_local(xp, zp, geo.L)
        u_loc, w_loc = strength * ud, strength * wd
    ug, wg = geo.to_global(u_loc, w_loc)
    u = U * np.cos(alpha) + ug.sum(axis=1)
    w = U * np.sin(alpha) + wg.sum(axis=1)
    if method == "doublet":
        uw, ww = _wake_velocity(px, pz, geo.x[0], geo.z[0])
        u, w = u + sol["gamma"] * uw, w + sol["gamma"] * ww
    return u, w


def _chunked_velocity(geo, method, sol, U, alpha, px, pz, chunk: int = 4000):
    """velocity_field in point blocks to bound the (points x panels) temporaries."""
    parts = [velocity_field(geo, method, sol, U, alpha, px[i:i + chunk], pz[i:i + chunk])
             for i in range(0, len(px), chunk)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _trace_streamlines(geo, method, sol, U, alpha, bounds, count, ds, max_steps):
    x_min, x_max, z_min, z_max = bounds
    t = np.linspace(-1.0, 1.0, count)
    z0 = 0.5 * (z_max - z_min) * 0.9 * np.sinh(2.5 * t) / np.sinh(2.5) + 0.5 * (z_max + z_min)
    pos = np.column_stack((np.full(count, x_min), z0))
    active = np.ones(count, dtype=bool)
    paths = [[pos[i].copy()] for i in range(count)]

    def direction(p):
        u, w = velocity_field(geo, method, sol, U, alpha, p[:, 0], p[:, 1])
        mag = np.maximum(np.hypot(u, w), _EPS)
        return np.column_stack((u / mag, w / mag))

    for _ in range(max_steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        p = pos[idx]
        mid = p + 0.5 * ds * direction(p)
        new = p + ds * direction(mid)
        stop = ((new[:, 0] > x_max) | (new[:, 1] < z_min) | (new[:, 1] > z_max)
                | geo.inside(new[:, 0], new[:, 1]))
        for k, i in enumerate(idx):
            if stop[k]:
                active[i] = False
            else:
                pos[i] = new[k]
                paths[i].append(new[k].copy())
    return [np.asarray(pth) for pth in paths if len(pth) > 2]


def _effective_cond(A: np.ndarray, method: str) -> float:
    """Condition number; for doublets the exact null direction is excluded."""
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / (s[-2] if method == "doublet" else s[-1]))


def _arr(a, decimals: int = 6) -> list:
    a = np.asarray(a, dtype=float)
    out = np.round(a, decimals)
    if out.ndim == 1:
        return [float(v) if np.isfinite(v) else None for v in out]
    return [_arr(row, decimals) for row in out]


def solve_airfoil_panels(code: str, chord: float, alpha_deg: float, U: float,
                         n_panels: int, method: str, with_field: bool = True) -> dict[str, Any]:
    """Solve the panel problem and return plain-Python data for plotting."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if not (len(code) == 4 and code.isdigit()):
        raise ValueError("NACA must be a four-digit code")
    if chord <= 0 or U <= 0:
        raise ValueError("chord and velocity must be positive")
    n_panels = int(n_panels) // 2 * 2
    if not 20 <= n_panels <= 400:
        raise ValueError("n_panels must be between 20 and 400")

    alpha = np.radians(alpha_deg)
    x, z = naca4_panel_nodes(code, chord, n_panels)
    geo = PanelGeometry(x, z, chord)

    sol = (_solve_source if method == "source" else _solve_doublet)(geo, U, alpha)
    Vt = sol["Vt"]
    cp = 1.0 - (Vt / U) ** 2
    if not np.all(np.isfinite(cp)):
        raise ValueError("Panel solution is not finite; try a different panel count")

    cx = -np.sum(cp * geo.n_x * geo.L) / chord
    cz = -np.sum(cp * geo.n_z * geo.L) / chord
    cl_pressure = -cx * np.sin(alpha) + cz * np.cos(alpha)
    cd_pressure = cx * np.cos(alpha) + cz * np.sin(alpha)
    cm = np.sum(cp * (geo.n_z * (geo.x_mid - 0.25 * chord) - geo.n_x * geo.z_mid) * geo.L) / chord**2
    if method == "doublet":
        cl = float(-2.0 * sol["gamma"] / (U * chord))
        cl_kutta = cl
    else:
        cl = float(cl_pressure)
        cl_kutta = float(2.0 * sol["gamma"] * geo.L.sum() / (U * chord))

    lower = np.arange(0, geo.i_le)
    upper = np.arange(geo.i_le, geo.n)
    result: dict[str, Any] = {
        "method": method,
        "naca": code,
        "n_panels": geo.n,
        "alpha": alpha_deg,
        "coefficients": {
            "cl": float(cl), "cl_pressure": float(cl_pressure), "cl_kutta": float(cl_kutta),
            "cd_pressure": float(cd_pressure), "cm_quarter": float(cm),
            "cp_min": float(cp.min()), "cp_max": float(cp.max()),
        },
        "diagnostics": {
            "cond": _effective_cond(sol["A"], method),
            "bc_residual": sol["bc_residual"],
            "kutta_residual": None if np.isnan(sol["kutta_residual"]) else sol["kutta_residual"],
        },
        "geometry": {
            "x": _arr(geo.x), "z": _arr(geo.z),
            "x_mid": _arr(geo.x_mid), "z_mid": _arr(geo.z_mid),
            "n_x": _arr(geo.n_x), "n_z": _arr(geo.n_z), "length": _arr(geo.L),
        },
        "distribution": {
            "cp": _arr(cp), "vt": _arr(Vt), "strength": _arr(sol["strength"]),
            "gamma": float(sol["gamma"]),
            "upper": upper.tolist(), "lower": lower.tolist(),
        },
    }

    if with_field:
        x_min, x_max = -0.4 * chord, 1.6 * chord
        z_min, z_max = -0.6 * chord, 0.6 * chord
        gx = np.linspace(x_min, x_max, 320)
        gz = np.linspace(z_min, z_max, 160)
        PX, PZ = np.meshgrid(gx, gz)
        px, pz = PX.ravel(), PZ.ravel()
        u, w = _chunked_velocity(geo, method, sol, U, alpha, px, pz)
        cp_field = 1.0 - (u**2 + w**2) / U**2
        cp_field[geo.inside(px, pz)] = np.nan
        streams = _trace_streamlines(
            geo, method, sol, U, alpha, (x_min, x_max, z_min, z_max),
            count=41, ds=0.01 * chord, max_steps=int(2.0 * (x_max - x_min) / (0.01 * chord)),
        )
        stream_data = []
        for s in streams:
            us, ws = velocity_field(geo, method, sol, U, alpha, s[:, 0], s[:, 1])
            stream_data.append({"x": _arr(s[:, 0], 5), "z": _arr(s[:, 1], 5),
                                "speed": _arr(np.hypot(us, ws) / U, 3)})
        result["field"] = {
            "x": _arr(gx, 5), "z": _arr(gz, 5),
            "cp": _arr(cp_field.reshape(PX.shape), 4),
            "streamlines": stream_data,
        }
    return result
