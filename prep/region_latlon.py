"""Cell-centre lat/lon of ne{N}pg2 cells for a sub-region of one cube face.

Vectorised re-statement of the official pure-Python TempestRemap port in
third_party/STRATA/data_prep/scrip_generation (generate_exodus_meshes.py +
scrip_convert.py + derive_latlon.py), restricted to the cells we need, so that
the 2.3 GB global SCRIP file never has to be built on a laptop.

The floating-point operation order of the official code is kept on purpose:
`validate_region_latlon.py` checks this module against the official scripts for
every cell of a small grid (bit-identical float32 is the acceptance criterion).

Column ordering (same as the SCRIP / Zarr ``ncol`` ordering):
    ncol = ((face * ne + j) * ne + i) * npg*npg + q * npg + p
and the model's 2-D face image has x = j*npg + q, y = i*npg + p
(see ScreamDataSource._compute_flat_indices in screamcast/earth2studio_wrappers.py).
"""

from __future__ import annotations

import math

import numpy as np

_INV = 1.0 / math.sqrt(3.0)
_CORNERS = np.array(
    [
        (_INV, -_INV, -_INV),
        (_INV, _INV, -_INV),
        (-_INV, _INV, -_INV),
        (-_INV, -_INV, -_INV),
        (_INV, -_INV, _INV),
        (_INV, _INV, _INV),
        (-_INV, _INV, _INV),
        (-_INV, -_INV, _INV),
    ],
    dtype=np.float64,
)

# (a, b) corner pairs of the 12 cube edges, in the order of generate_cs_mesh.
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)]
# Per face: (edge1, reversed1, edge2, reversed2, edge3, reversed3) as passed to
# _generate_faces_from_quad (edge0 is unused when halo == 0).
_FACES = [
    (4, False, 5, False, 8, False),
    (5, False, 6, False, 9, False),
    (6, False, 7, False, 10, False),
    (7, False, 4, False, 11, False),
    (3, False, 1, True, 0, False),
    (11, True, 9, False, 10, True),
]


def _normalize(v: np.ndarray) -> np.ndarray:
    # math.sqrt(x*x + y*y + z*z) in the same summation order as _normalize().
    r = np.sqrt(v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1] + v[..., 2] * v[..., 2])
    return v / r[..., None]


def _alpha(n: int, k: np.ndarray) -> np.ndarray:
    a = k / n
    return 0.5 * (np.tan(0.25 * math.pi * (2.0 * a - 1.0)) + 1.0)


def _edge_points(n: int, p0: np.ndarray, p1: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """Points k (0..n) of _generate_cs_multiedge_vertices(p0 -> p1).

    p0/p1 may be (3,) or broadcastable (..., 3) arrays.
    """
    ks = np.asarray(ks)
    a = _alpha(n, ks.astype(np.float64))[..., None]
    pts = _normalize(p0 + (p1 - p0) * a)
    out = np.where((ks == 0)[..., None], np.broadcast_to(p0, pts.shape), pts)
    out = np.where((ks == n)[..., None], np.broadcast_to(p1, pts.shape), out)
    return out


def _cube_edge(n: int, e: int, rev: bool, ks: np.ndarray) -> np.ndarray:
    a, b = _EDGES[e]
    ks = np.asarray(ks)
    k_src = (n - ks) if rev else ks
    return _edge_points(n, _CORNERS[a], _CORNERS[b], k_src)


def _row_nodes(n: int, face: int, js: np.ndarray, iks: np.ndarray) -> np.ndarray:
    """Node (row js, col iks) of a face: shape (len(js), len(iks), 3).

    Row j (0..n) is the multiedge between edge1[j] and edge2[j]; row 0 is
    generated from edge1[0], edge2[0]; row n is edge3 itself.
    """
    e1, r1, e2, r2, e3, r3 = _FACES[face]
    js = np.asarray(js)
    iks = np.asarray(iks)
    left = _cube_edge(n, e1, r1, js)  # (J, 3)
    right = _cube_edge(n, e2, r2, js)
    rows = _edge_points(n, left[:, None, :], right[:, None, :], iks[None, :])
    top = js == n
    if np.any(top):
        rows[top] = _cube_edge(n, e3, r3, iks)[None]
    return rows


def _bilinear(n0, n1, n2, n3, d_a: float, d_b: float) -> np.ndarray:
    v = (
        (1.0 - d_a) * (1.0 - d_b) * n0
        + d_a * (1.0 - d_b) * n1
        + d_a * d_b * n2
        + (1.0 - d_a) * d_b * n3
    )
    return _normalize(v)


def _xyz_to_lonlat_deg(x, y, z):
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z))
    return np.mod(lon, 360.0), lat


def face_image_latlon(
    ne: int, npg: int, face: int, x0: int, nx: int, y0: int, ny: int
) -> tuple[np.ndarray, np.ndarray]:
    """float32 (lat, lon) in degrees for face-image pixels [x0:x0+nx, y0:y0+ny]."""
    xs = np.arange(x0, x0 + nx)
    ys = np.arange(y0, y0 + ny)
    j = xs // npg
    q = xs % npg
    i = ys // npg
    p = ys % npg
    juniq = np.unique(np.concatenate([j, j + 1]))
    iuniq = np.unique(np.concatenate([i, i + 1]))
    nodes = _row_nodes(ne, face, juniq, iuniq)  # (J, I, 3)
    jj = np.searchsorted(juniq, j)[:, None]
    ii = np.searchsorted(iuniq, i)[None, :]
    bot0 = nodes[jj, ii]  # node0 = bot[i]
    bot1 = nodes[jj, ii + 1]  # node1 = bot[i+1]
    top1 = nodes[jj + 1, ii + 1]  # node2 = top[i+1]
    top0 = nodes[jj + 1, ii]  # node3 = top[i]

    # official: d_w = [1/np]*np, cumulative sums
    acc = [0.0]
    for _ in range(npg):
        acc.append(acc[-1] + 1.0 / npg)
    acc = np.array(acc)
    pp = p[None, :]
    qq = q[:, None]
    corners = []
    for c in range(4):
        px = pp + ((c + 1) // 2) % 2
        qx = qq + (c // 2)
        d_a = acc[px][..., None]
        d_b = acc[qx][..., None]
        corners.append(_bilinear(bot0, bot1, top1, top0, d_a, d_b))
    xv = np.stack([c[..., 0] for c in corners], axis=-1)
    yv = np.stack([c[..., 1] for c in corners], axis=-1)
    zv = np.stack([c[..., 2] for c in corners], axis=-1)
    cx = xv.mean(axis=-1)
    cy = yv.mean(axis=-1)
    cz = zv.mean(axis=-1)
    r = np.sqrt(cx * cx + cy * cy + cz * cz)
    lon, lat = _xyz_to_lonlat_deg(cx / r, cy / r, cz / r)
    return lat.astype(np.float32), lon.astype(np.float32)


def face_centers_coarse(ne: int, npg: int, face: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Cheap coarse lat/lon of a face (every `stride` pixels), for locating a city."""
    n = ne * npg
    xs = np.arange(0, n, stride)
    lat = np.empty((len(xs), len(xs)), np.float32)
    lon = np.empty_like(lat)
    for a, x in enumerate(xs):
        la, lo = face_image_latlon(ne, npg, face, int(x), 1, 0, n)
        lat[a] = la[0, ::stride]
        lon[a] = lo[0, ::stride]
    return lat, lon


def locate(lat_deg: float, lon_deg: float, ne: int = 1024, npg: int = 2) -> tuple[int, int, int]:
    """(face, x, y) of the cell whose centre is nearest to (lat, lon)."""
    target = np.array(
        [
            math.cos(math.radians(lat_deg)) * math.cos(math.radians(lon_deg)),
            math.cos(math.radians(lat_deg)) * math.sin(math.radians(lon_deg)),
            math.sin(math.radians(lat_deg)),
        ]
    )
    best = None
    n = ne * npg
    for face in range(6):
        la, lo = face_centers_coarse(ne, npg, face, 32)
        la, lo = np.radians(la.astype(np.float64)), np.radians(lo.astype(np.float64))
        v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
        d = v @ target
        k = np.unravel_index(np.argmax(d), d.shape)
        if best is None or d[k] > best[0]:
            best = (d[k], face, k[0] * 32, k[1] * 32)
    _, face, cx, cy = best
    # refine in a window around the coarse hit
    x0, y0 = max(cx - 48, 0), max(cy - 48, 0)
    nx, ny = min(96, n - x0), min(96, n - y0)
    la, lo = face_image_latlon(ne, npg, face, x0, nx, y0, ny)
    la, lo = np.radians(la.astype(np.float64)), np.radians(lo.astype(np.float64))
    v = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    k = np.unravel_index(np.argmax(v @ target), la.shape)
    return face, x0 + int(k[0]), y0 + int(k[1])
