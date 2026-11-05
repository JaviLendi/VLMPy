#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Javier Lendínez Castillo
# SPDX-License-Identifier: AGPL-3.0-only

"""
STL export of the complete plane (wing, horizontal and vertical stabilizers).

Conventions (identical to lib.geometry.calculate_geometry):
- x streamwise (aft), y spanwise (right), z up; lengths in m, angles in rad.
- Wing and horizontal stabilizer are symmetric: each section's `span_fraction`
  is the TOTAL span of that section (half per side). The vertical stabilizer
  is a single (non-mirrored) surface whose span runs along +z.
- A section's root chord/station is the previous section's tip (as in the VLM);
  `twist` is incremental per section and accumulates from the root, rotating
  the profile about its quarter chord (positive = leading edge up).
- `alpha` of a stabilizer is a rigid rotation about the y axis through the
  surface root before translation, as done by the VLM geometry. The flight
  angle of attack is NOT applied: the STL is the body geometry.
- Each surface is exported as a closed solid; solids are not boolean-merged.
- Flap/aileron deflections are not represented.
"""

import numpy as np
from stl import mesh, Mode

from lib.naca import naca_airfoil

_REQUIRED_KEYS = ('span_fraction', 'chord_root', 'chord_tip', 'sweep', 'NACA_root', 'NACA_tip')


def _validate_section(sec, label, default_dihedral=None):
    if not isinstance(sec, dict):
        raise ValueError(f"{label} must be a dict, got {type(sec)}")
    for key in _REQUIRED_KEYS:
        if key not in sec:
            raise ValueError(f"{label} missing required key '{key}'")
    if 'dihedral' not in sec and default_dihedral is None:
        raise ValueError(f"{label} missing required key 'dihedral'")
    for key in ('span_fraction', 'chord_root', 'chord_tip', 'sweep', 'dihedral', 'twist'):
        if key in sec and not np.isfinite(float(sec[key])):
            raise ValueError(f"{label} key '{key}' must be finite")
    if float(sec['span_fraction']) <= 0:
        raise ValueError(f"{label} 'span_fraction' must be > 0")
    if float(sec['chord_root']) <= 0 or float(sec['chord_tip']) <= 0:
        raise ValueError(f"{label} chords must be > 0")
    for key in ('NACA_root', 'NACA_tip'):
        code = str(sec[key])
        if len(code) != 4 or not code.isdigit():
            raise ValueError(f"{label} '{key}' must be a 4-digit NACA code, got '{code}'")


def _rot_y(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _right_side_groups(sections, symmetric, label, default_dihedral=None):
    """Station lists (continuous lofts) for one side, starting at the root."""
    groups, current = [], None
    y = z = x_le = 0.0
    twist_acc = 0.0
    prev_chord = prev_naca = None
    for i, sec in enumerate(sections):
        _validate_section(sec, f"{label} section {i}", default_dihedral)
        dihedral = float(sec.get('dihedral', default_dihedral))
        span = float(sec['span_fraction'])
        span2 = span / 2 if symmetric else span
        root_chord = float(sec['chord_root']) if prev_chord is None else prev_chord
        root_naca = str(sec['NACA_root'])

        if current is None or root_naca != prev_naca:
            current = [dict(le=np.array([x_le, y, z]), chord=root_chord, naca=root_naca, twist=twist_acc)]
            groups.append(current)

        y += span2 * np.cos(dihedral)
        z += span2 * np.sin(dihedral)
        x_le += np.tan(float(sec['sweep'])) * span2 * (np.cos(dihedral) if symmetric else 1.0)
        twist_acc += float(sec.get('twist', 0.0))
        prev_chord = float(sec['chord_tip'])
        prev_naca = str(sec['NACA_tip'])
        current.append(dict(le=np.array([x_le, y, z]), chord=prev_chord, naca=prev_naca, twist=twist_acc))
    return groups


def _mirror(station):
    le = station['le'].copy()
    le[1] = -le[1]
    return dict(station, le=le)


def _profile(naca, chord, twist, resolution):
    """Closed airfoil ring in (x, z): lower TE -> LE -> upper TE (TE gap closes by wrap)."""
    x, z_upper, z_lower, *_ = naca_airfoil(naca, chord, 0.0, resolution)
    xs = np.concatenate([x[::-1], x[1:]])
    zs = np.concatenate([z_lower[::-1], z_upper[1:]])
    if np.hypot(xs[0] - xs[-1], zs[0] - zs[-1]) < 1e-9 * chord:
        xs, zs = xs[:-1], zs[:-1]  # sharp TE: avoid a duplicated vertex
    if twist:
        xq = 0.25 * chord
        c, s = np.cos(twist), np.sin(twist)
        xs, zs = c * (xs - xq) + s * zs + xq, -s * (xs - xq) + c * zs
    return xs, zs


def _loft_solid(stations, x_axis, up, rotation, shift, resolution):
    """Closed, outward-oriented triangle soup (K, 3, 3) lofted through the stations."""
    rings = []
    for st in stations:
        xs, zs = _profile(st['naca'], st['chord'], st['twist'], resolution)
        le = rotation @ st['le'] + shift
        rings.append(le[None, :] + xs[:, None] * x_axis + zs[:, None] * up)
    rings = np.asarray(rings)

    tris = []
    for a, b in zip(rings[:-1], rings[1:]):
        a1, b1 = np.roll(a, -1, axis=0), np.roll(b, -1, axis=0)
        tris.append(np.stack([a, a1, b], axis=1))
        tris.append(np.stack([a1, b1, b], axis=1))
    for ring, flip in ((rings[0], True), (rings[-1], False)):
        centre = np.broadcast_to(ring.mean(axis=0), ring.shape)
        nxt = np.roll(ring, -1, axis=0)
        tris.append(np.stack([centre, nxt, ring] if flip else [centre, ring, nxt], axis=1))
    tris = np.concatenate(tris)

    volume = np.einsum('ij,ij->', tris[:, 0], np.cross(tris[:, 1], tris[:, 2])) / 6.0
    if volume < 0:
        tris = tris[:, [0, 2, 1], :]
    return tris


def _surface_triangles(sections, symmetric, label, alpha=0.0, translate=(0.0, 0.0),
                       up=(0.0, 0.0, 1.0), default_dihedral=None, resolution=80):
    groups = _right_side_groups(sections, symmetric, label, default_dihedral)
    lofts = []
    for index, group in enumerate(groups):
        if not symmetric:
            lofts.append(group)
            continue
        mirrored = [_mirror(s) for s in group]
        if index == 0:
            lofts.append(mirrored[:0:-1] + group)
        else:
            lofts.extend([group, mirrored])

    rotation = _rot_y(float(alpha))
    x_axis = rotation @ np.array([1.0, 0.0, 0.0])
    up_vec = rotation @ np.asarray(up, dtype=float)
    shift = np.array([float(translate[0]), 0.0, float(translate[1])])
    return np.concatenate([_loft_solid(l, x_axis, up_vec, rotation, shift, resolution) for l in lofts])


def build_plane_mesh(plane, naca_resolution=80):
    """Build a numpy-stl mesh of the wing and, if defined, both stabilizers."""
    if not isinstance(plane, dict):
        raise ValueError("plane must be a dict")
    wing = plane.get('wing_sections')
    if not isinstance(wing, list) or not wing:
        raise ValueError("plane['wing_sections'] must be a non-empty list")

    parts = [_surface_triangles(wing, True, 'wing', resolution=naca_resolution)]

    hs = plane.get('horizontal_stabilizer')
    if hs:
        parts.append(_surface_triangles(
            [hs], True, 'horizontal stabilizer', alpha=hs.get('alpha', 0.0),
            translate=(hs.get('x_translate', 0.0), hs.get('z_translate', 0.0)),
            resolution=naca_resolution))

    vs = plane.get('vertical_stabilizer')
    if vs:
        parts.append(_surface_triangles(
            [vs], False, 'vertical stabilizer', alpha=vs.get('alpha', 0.0),
            translate=(vs.get('x_translate', 0.0), vs.get('z_translate', 0.0)),
            up=(0.0, 1.0, 0.0), default_dihedral=np.pi / 2, resolution=naca_resolution))

    tris = np.concatenate(parts)
    data = np.zeros(len(tris), dtype=mesh.Mesh.dtype)
    data['vectors'] = tris
    plane_mesh = mesh.Mesh(data)
    plane_mesh.update_normals()
    return plane_mesh


def _save(plane_mesh, output_filename, format):
    fmt = str(format).lower()
    if fmt not in ('binary', 'ascii'):
        raise ValueError("format must be 'binary' or 'ascii'")
    plane_mesh.save(str(output_filename), mode=Mode.BINARY if fmt == 'binary' else Mode.ASCII)


def wing_to_stl_with_naca_config(plane_config, output_filename='wing_with_profile.stl',
                                 naca_resolution=100, format='binary'):
    """Export a plane dict (wing + optional stabilizers) to STL and return the mesh."""
    plane_mesh = build_plane_mesh(plane_config, naca_resolution)
    _save(plane_mesh, output_filename, format)
    return plane_mesh


def wing_to_stl_with_naca_profile(vlm_object, output_filename='wing_with_profile.stl',
                                  naca_resolution=50, format='binary'):
    """Export the plane held by a VLM object to STL and return the mesh."""
    return wing_to_stl_with_naca_config(vlm_object.plane, output_filename, naca_resolution, format)
