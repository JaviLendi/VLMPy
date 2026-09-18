"""JSON-ready geometry serializers for the interactive frontend."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _array_points(points: Iterable[Any], dimensions: int = 3) -> list[list[float]]:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != dimensions:
        raise ValueError(f"Expected an array with shape (N, {dimensions})")
    if not np.isfinite(array).all():
        raise ValueError("Geometry contains non-finite coordinates")
    return array.tolist()


def serialize_airfoil(x: Iterable[float], z: Iterable[float], *, units: str = "m") -> dict[str, Any]:
    x_array = np.asarray(x, dtype=float).reshape(-1)
    z_array = np.asarray(z, dtype=float).reshape(-1)
    if x_array.shape != z_array.shape or x_array.size < 2:
        raise ValueError("Airfoil coordinates must contain at least two matching points")
    if not np.isfinite(x_array).all() or not np.isfinite(z_array).all():
        raise ValueError("Airfoil contains non-finite coordinates")
    return {
        "units": units,
        "coordinates": {
            "x": x_array.tolist(),
            "z": z_array.tolist(),
        },
        "bounds": {
            "x_min": float(x_array.min()),
            "x_max": float(x_array.max()),
            "z_min": float(z_array.min()),
            "z_max": float(z_array.max()),
        },
    }


def serialize_panel_geometry(panel_data: list[Any]) -> dict[str, Any]:
    vertices: list[list[float]] = []
    indices: list[int] = []
    panel_ids: list[int] = []
    control_points: list[list[float]] = []
    normals: list[dict[str, list[float]]] = []

    for panel_index, item in enumerate(panel_data):
        panel = np.asarray(item[1], dtype=float)
        if panel.shape != (4, 3):
            raise ValueError("Each panel must contain four 3D vertices")
        offset = len(vertices)
        vertices.extend(_array_points(panel))
        indices.extend([offset, offset + 1, offset + 2, offset, offset + 2, offset + 3])
        panel_ids.extend([panel_index, panel_index])
        control_points.append(_array_points([item[3]])[0])
        normal_vector, midpoint = item[4]
        normals.append({
            "vector": _array_points([normal_vector])[0],
            "origin": _array_points([midpoint])[0],
        })

    return {
        "coordinate_system": "VLMPy",
        "units": "m",
        "vertices": vertices,
        "indices": indices,
        "panel_ids": panel_ids,
        "control_points": control_points,
        "normals": normals,
        "panel_count": len(panel_data),
    }
