"""Geometry-only fuselage meshes in aircraft coordinates and metres."""

import hashlib
import io
import struct

import numpy as np
from stl import mesh


MAX_STL_BYTES = 5 * 1024 * 1024
MAX_TRIANGLES = 50000


class Fuselage:
    """Geometry-only fuselage using the existing configuration dictionary."""

    def __init__(self, configuration):
        self.configuration = configuration

    def mesh(self):
        """Return vertices and triangular faces in aircraft coordinates in metres."""
        return fuselage_mesh(self.configuration)


def build_fuselage_mesh(stations, circumferential_points=32):
    """Return vertices and triangular faces for closed elliptical stations.

    Stations are mappings with x, y_radius and z_radius in metres. The
    centreline lies on the x axis; the first and last stations are points.
    """
    if not isinstance(stations, list) or not 3 <= len(stations) <= 500:
        raise ValueError('Between 3 and 500 fuselage stations are required')
    if not isinstance(circumferential_points, int) or isinstance(circumferential_points, bool) or not 4 <= circumferential_points <= 96:
        raise ValueError('circumferential_points must be an integer between 4 and 96')
    try:
        values = np.asarray([[station['x'], station['y_radius'], station['z_radius']] for station in stations], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Each station requires numeric x, y_radius and z_radius') from error
    if not np.isfinite(values).all():
        raise ValueError('Station values must be finite')
    if np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError('Station x coordinates must be strictly increasing')
    if np.any(values[[0, -1], 1:] != 0) or np.any(values[1:-1, 1:] <= 0):
        raise ValueError('End stations must have zero radii and interior radii must be positive')

    angles = 2 * np.pi * np.arange(circumferential_points) / circumferential_points
    vertices = [[values[0, 0], 0.0, 0.0]]
    for x_position, y_radius, z_radius in values[1:-1]:
        vertices.extend(np.column_stack((
            np.full(circumferential_points, x_position),
            y_radius * np.cos(angles), z_radius * np.sin(angles),
        )).tolist())
    vertices.append([values[-1, 0], 0.0, 0.0])
    faces = []
    for index in range(circumferential_points):
        next_index = (index + 1) % circumferential_points
        faces.append([0, 1 + next_index, 1 + index])
        for ring in range(len(values) - 3):
            first = 1 + ring * circumferential_points
            second = first + circumferential_points
            faces.extend([[first + index, second + next_index, second + index],
                          [first + index, first + next_index, second + next_index]])
        last = 1 + (len(values) - 3) * circumferential_points
        faces.append([len(vertices) - 1, last + index, last + next_index])
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def inspect_stl(content, filename, units):
    """Parse an STL without modifying it and return JSON-ready geometry and diagnostics."""
    scales = {'m': 1.0, 'mm': 0.001, 'ft': 0.3048}
    if units not in scales:
        raise ValueError('Select STL units: m, mm or ft')
    if not isinstance(content, bytes) or not content or len(content) > MAX_STL_BYTES:
        raise ValueError('STL file must be nonempty and no larger than 5 MiB')
    is_binary = len(content) >= 84 and 84 + 50 * struct.unpack('<I', content[80:84])[0] == len(content)
    try:
        source = mesh.Mesh.from_file(filename, fh=io.BytesIO(content), calculate_normals=False)
        triangles = np.asarray(source.vectors, dtype=float)
        source_normals = np.asarray(source.normals, dtype=float)
    except Exception as error:
        raise ValueError('Invalid or unsupported STL file') from error
    if len(triangles) == 0 or len(triangles) > MAX_TRIANGLES:
        raise ValueError('STL must contain between 1 and 50000 triangles')
    if not np.isfinite(triangles).all() or not np.isfinite(source_normals).all():
        raise ValueError('STL contains non-finite coordinates or normals')
    triangles = triangles * scales[units]
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    degenerate = np.linalg.norm(normals, axis=1) == 0
    if degenerate.any():
        raise ValueError('STL contains degenerate triangles')
    normal_lengths = np.linalg.norm(source_normals, axis=1)
    supplied = normal_lengths > 0
    disagree = np.einsum('ij,ij->i', normals, source_normals) < 0
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    unique_edges, edge_inverse, edge_counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    edge_direction = np.bincount(edge_inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    parents = np.arange(len(vertices))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in unique_edges:
        parents[root(first)] = root(second)
    roots = [root(index) for index in range(len(vertices))]
    component_roots = list(dict.fromkeys(roots[index] for index in faces[:, 0]))
    root_to_component = {value: index for index, value in enumerate(component_roots)}
    component_ids = np.asarray([root_to_component[roots[index]] for index in faces[:, 0]], dtype=int)
    components = []
    for component_index in range(len(component_roots)):
        component_triangles = triangles[component_ids == component_index]
        components.append({'triangle_count': len(component_triangles),
                           'bounds_m': [component_triangles.min(axis=(0, 1)).tolist(),
                                        component_triangles.max(axis=(0, 1)).tolist()]})
    report = {
        'format': 'binary' if is_binary else 'ascii',
        'triangle_count': len(faces),
        'vertex_count': len(vertices),
        'duplicate_vertices': int(len(inverse) - len(vertices)),
        'connected_components': len(components),
        'components': components,
        'boundary_edges': int(np.count_nonzero(edge_counts == 1)),
        'nonmanifold_edges': int(np.count_nonzero(edge_counts > 2)),
        'inconsistent_edges': int(np.count_nonzero((edge_counts == 2) & (edge_direction != 0))),
        'missing_normals': int(np.count_nonzero(~supplied)),
        'opposed_normals': int(np.count_nonzero(supplied & disagree)),
        'bounds_m': [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }
    return {
        'type': 'stl', 'vertices': vertices.tolist(), 'faces': faces.tolist(),
        'component_ids': component_ids.tolist(), 'selected_component': None,
        'source': {'filename': filename, 'units': units, 'sha256': hashlib.sha256(content).hexdigest(),
                   'format': report['format']},
    }, report


def fuselage_mesh(configuration):
    """Return a fuselage mesh transformed to aircraft coordinates in metres.

    STL geometry is scaled about its source origin, then rotated in degrees
    about fixed x, y, then z axes, followed by translation in metres.
    """
    if not isinstance(configuration, dict):
        raise ValueError('Fuselage configuration must be an object')
    if configuration.get('type') == 'stations':
        vertices, faces = build_fuselage_mesh(configuration.get('stations'))
    elif configuration.get('type') == 'stl':
        try:
            vertices = np.asarray(configuration['vertices'], dtype=float)
            raw_faces = np.asarray(configuration['faces'])
            faces = np.asarray(raw_faces, dtype=int)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError('Invalid fuselage mesh') from error
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
                or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
                or len(faces) > MAX_TRIANGLES or not np.isfinite(vertices).all()
                or not np.array_equal(raw_faces, faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Invalid fuselage mesh')
        selected = configuration.get('selected_component')
        if selected is not None:
            component_ids = np.asarray(configuration.get('component_ids', []))
            if (not isinstance(selected, int) or isinstance(selected, bool) or selected < 0
                    or component_ids.shape != (len(faces),) or not np.any(component_ids == selected)):
                raise ValueError('Invalid fuselage component selection')
            faces = faces[component_ids == selected]
            used_vertices, inverse = np.unique(faces, return_inverse=True)
            vertices, faces = vertices[used_vertices], inverse.reshape(-1, 3)
        triangles = vertices[faces]
        if np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) == 0):
            raise ValueError('Fuselage mesh contains degenerate triangles')
    else:
        raise ValueError('Fuselage type must be stations or stl')
    try:
        translation = np.asarray(configuration.get('translation', [0, 0, 0]), dtype=float)
        rotation = np.asarray(configuration.get('rotation_deg', [0, 0, 0]), dtype=float)
        scale = float(configuration.get('scale', 1.0)) if configuration.get('type') == 'stl' else 1.0
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid fuselage placement') from error
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('STL scale must be a positive finite number')
    if translation.shape != (3,) or rotation.shape != (3,) or not np.isfinite(translation).all() or not np.isfinite(rotation).all():
        raise ValueError('Fuselage placement requires finite three-component vectors')
    roll, pitch, yaw = np.radians(rotation)
    cx, cy, cz = np.cos([roll, pitch, yaw])
    sx, sy, sz = np.sin([roll, pitch, yaw])
    rotate_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rotate_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rotate_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    with np.errstate(over='ignore', invalid='ignore'):
        transformed = (vertices * scale) @ (rotate_z @ rotate_y @ rotate_x).T + translation
    if not np.isfinite(transformed).all():
        raise ValueError('Fuselage placement produces non-finite coordinates')
    return transformed, faces