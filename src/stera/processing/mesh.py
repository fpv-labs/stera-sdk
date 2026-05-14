"""MeshRefiner: clean, densify, colorize, and de-clutter SLAM meshes.

Single class that bundles the pipeline:

  1. ``clean``    — pymeshlab cascade (dedup, repair, remove floaters, close
                    small holes, Laplacian smoothing).
  2. ``densify``  — Loop subdivision (×4 faces per iteration).
  3. ``colorize`` — per-vertex texture mapping with bilinear sampling,
                    view-angle + 1/depth² weighting, and a depth-buffer
                    occlusion test.
  4. ``brighten`` — lift dim pixels so meshes from low-light recordings
                    are still readable.
  5. ``strip_table_clutter`` — RANSAC-detect the kitchen-counter plane and
                    remove anything sitting on it. Walls attached to the
                    counter are protected by kd-tree proximity.
  6. ``fill_table`` — drop a flat triangulated patch on the counter plane,
                    region defined by a 2D occupancy mask of inlier density
                    (so it follows L-shaped / concave counters instead of
                    over-extending past the actual surface).

``refine()`` runs whichever stages are enabled by the constructor flags and
returns a single ``RefinedMesh``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from stera.data import MCAPReader

logger = logging.getLogger(__name__)


def _require(module: str, what: str):
    """Import an optional dependency or raise a clear install hint."""
    try:
        return __import__(module)
    except ImportError as e:
        raise ImportError(
            f"{what} requires the `mesh` extra. Install with: "
            f"pip install 'stera-sdk[mesh]'"
        ) from e


# ---------------------------------------------------------------- public utils

def clean_mesh_by_edge_length(
    verts: np.ndarray,
    faces: np.ndarray,
    max_edge_len: float = 0.15,
) -> np.ndarray:
    """Drop triangles with any edge longer than ``max_edge_len`` or zero area.

    Cheap geometric filter used as a baseline cleanup when pymeshlab isn't
    available or wanted. Returns a filtered ``faces`` array; ``verts`` is not
    touched.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e0 = np.linalg.norm(v1 - v0, axis=1)
    e1 = np.linalg.norm(v2 - v1, axis=1)
    e2 = np.linalg.norm(v0 - v2, axis=1)
    keep = (e0 < max_edge_len) & (e1 < max_edge_len) & (e2 < max_edge_len)
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    keep &= areas > 1e-8
    return faces[keep]


def brighten_colors(
    colors: np.ndarray,
    factor: float = 1.4,
    min_brightness: int = 60,
) -> np.ndarray:
    """Scale brightness then lift darks to ``min_brightness``. Returns uint8."""
    c = colors.astype(np.float32) * factor
    c = np.maximum(c, min_brightness)
    return np.clip(c, 0, 255).astype(np.uint8)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals as the area-weighted average of face normals."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    vn = np.zeros_like(verts)
    np.add.at(vn, faces[:, 0], face_normals)
    np.add.at(vn, faces[:, 1], face_normals)
    np.add.at(vn, faces[:, 2], face_normals)
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return (vn / norms).astype(np.float32)


def _resolve_color_speed(speed: float) -> tuple[int, Optional[int], bool]:
    """Map a [0, 1] quality dial to (every_n, max_frames, use_occlusion).

    - ``0.0`` — draft: every 30th frame, hard cap 400 frames, no occlusion test
      (fastest preview).
    - ``0.5`` — balanced: every ~5 frames, no cap, occlusion on (default).
    - ``1.0`` — full quality: every frame, no cap, occlusion on (slowest).

    ``every_n`` interpolates geometrically (30**(1-q)), so the dial is roughly
    log-linear in cost.
    """
    try:
        q = float(speed)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"color_speed must be a float in [0, 1], got {speed!r}"
        ) from e
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"color_speed={q}: must be in [0, 1]")
    every_n = max(1, round(30.0 ** (1.0 - q)))
    max_frames = 400 if q < 0.15 else None
    use_occlusion = q >= 0.5
    return every_n, max_frames, use_occlusion


# ---------------------------------------------------------------- result type

@dataclass
class RefinedMesh:
    """Output of :meth:`MeshRefiner.refine`."""

    vertices: np.ndarray         # (N, 3) float32 world-frame
    faces: np.ndarray            # (M, 3) int32
    vertex_colors: Optional[np.ndarray]   # (N, 3) uint8 or None
    vertex_normals: np.ndarray   # (N, 3) float32


# ---------------------------------------------------------------- internals

def _bilinear_sample(rgb: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear RGB lookup at float (u, v) image coords. Returns (N, 3) float."""
    h, w = rgb.shape[:2]
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1
    fu = (u - u0).astype(np.float32)
    fv = (v - v0).astype(np.float32)

    u0c = np.clip(u0, 0, w - 1); u1c = np.clip(u1, 0, w - 1)
    v0c = np.clip(v0, 0, h - 1); v1c = np.clip(v1, 0, h - 1)

    c00 = rgb[v0c, u0c].astype(np.float32)
    c10 = rgb[v0c, u1c].astype(np.float32)
    c01 = rgb[v1c, u0c].astype(np.float32)
    c11 = rgb[v1c, u1c].astype(np.float32)

    w00 = (1 - fu) * (1 - fv)
    w10 = fu       * (1 - fv)
    w01 = (1 - fu) * fv
    w11 = fu       * fv
    return (w00[:, None] * c00 + w10[:, None] * c10
            + w01[:, None] * c01 + w11[:, None] * c11)


def _plane_basis(plane_eq):
    """Return (origin, u, v, n) — orthonormal in-plane basis + plane normal."""
    a, b, c, d = plane_eq
    n = np.array([a, b, c], dtype=np.float64)
    n /= (np.linalg.norm(n) + 1e-12)
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    p0 = -d * n
    return p0, u, v, n


# ---------------------------------------------------------------- main class

class MeshRefiner:
    """Clean / densify / colorize / de-clutter a SLAM mesh from an MCAP session.

    Construct once per session and call :meth:`refine` to get a single
    ``RefinedMesh``. Individual stages (:meth:`clean`, :meth:`densify`,
    :meth:`colorize`, :meth:`strip_table_clutter`, :meth:`fill_table`) are
    public so callers can build custom pipelines.

    Heavy deps:
      • ``pymeshlab`` (cleanup, subdivision)
      • ``pyransac3d`` + ``scipy`` + ``matplotlib`` (clutter strip, fill)

    These are imported lazily; you can use the colorize-only path without
    pymeshlab/pyransac3d installed.

    Speed of the colorize stage scales linearly with the number of camera
    frames touched. Tune via:

      • ``color_speed`` — float in ``[0, 1]`` (default ``0.5``). 0 is the
        fastest draft (every 30th frame, no occlusion test); 1 is full
        quality (every frame, occlusion on). Geometric interpolation:
        ``every_n = round(30 ** (1 - q))``.
      • ``color_every_n`` — explicit override; sample every Nth frame.
      • ``color_max_frames`` — hard cap on frames used (None = no cap).
      • ``color_use_occlusion`` — toggle the depth-buffer occlusion test
        (off is ~30 % faster but lets walls bleed colour through one another).

    Reference timings on a 155k-vert / 11k-frame mcap:

        speed=0.0  →   ~31 s    (draft, capped to 400 frames)
        speed=0.5  →   ~37 s    (balanced, every 5 frames, occlusion)
        speed=1.0  →  ~100 s    (full quality)
    """

    def __init__(
        self,
        session: "MCAPReader",
        *,
        # --- cleanup
        cleanup: bool = True,
        min_component_diag_pct: float = 15.0,
        min_component_faces: int = 2000,
        max_hole_size: int = 30,
        laplacian_iters: int = 1,
        # --- densify
        subdivision_iters: int = 2,
        # --- colorize
        colorize: bool = True,
        color_speed: float = 0.5,
        color_every_n: Optional[int] = None,
        color_max_frames: Optional[int] = None,
        color_use_occlusion: Optional[bool] = None,
        min_view_angle_cos: float = 0.05,
        max_color_depth: float = 6.0,
        occlusion_tolerance: float = 0.08,
        brighten: bool = True,
        brighten_factor: float = 1.4,
        brighten_min: int = 60,
        # --- clutter strip (optional)
        strip_table_clutter: bool = False,
        world_up: np.ndarray = np.array([0.0, 1.0, 0.0]),
        ransac_dist_thresh: float = 0.03,
        ransac_min_inliers_pct: float = 1.5,
        ransac_max_planes: int = 15,
        horizontal_cos: float = 0.85,
        vertical_cos: float = 0.30,
        wall_min_inliers: int = 300,
        wall_protect_radius: float = 0.06,
        clutter_height_min: float = 0.01,
        clutter_height_max: float = 0.80,
        clutter_pad_factor: float = 1.05,
        # --- flat fill (optional, only meaningful with strip_table_clutter)
        fill_table: bool = False,
        fill_grid_res: float = 0.03,
        fill_occ_cell: float = 0.05,
        fill_occ_min_count: int = 2,
        fill_close_iters: int = 2,
        fill_dilate_iters: int = 1,
        fill_near_existing: float = 0.05,
    ):
        self._session = session
        # cleanup
        self.cleanup = bool(cleanup)
        self.min_component_diag_pct = float(min_component_diag_pct)
        self.min_component_faces = int(min_component_faces)
        self.max_hole_size = int(max_hole_size)
        self.laplacian_iters = int(laplacian_iters)
        # densify
        self.subdivision_iters = int(subdivision_iters)
        # colorize
        self.colorize_enabled = bool(colorize)
        speed_n, speed_max, speed_occ = _resolve_color_speed(color_speed)
        # Explicit per-knob values override the dial.
        self.color_every_n = int(color_every_n) if color_every_n is not None else speed_n
        self.color_max_frames = (
            int(color_max_frames) if color_max_frames is not None else speed_max
        )
        self.color_use_occlusion = (
            bool(color_use_occlusion) if color_use_occlusion is not None else speed_occ
        )
        self.min_view_angle_cos = float(min_view_angle_cos)
        self.max_color_depth = float(max_color_depth)
        self.occlusion_tolerance = float(occlusion_tolerance)
        self.brighten = bool(brighten)
        self.brighten_factor = float(brighten_factor)
        self.brighten_min = int(brighten_min)
        # clutter strip
        self.strip_table_clutter_enabled = bool(strip_table_clutter)
        self.world_up = np.asarray(world_up, dtype=np.float64)
        self.ransac_dist_thresh = float(ransac_dist_thresh)
        self.ransac_min_inliers_pct = float(ransac_min_inliers_pct)
        self.ransac_max_planes = int(ransac_max_planes)
        self.horizontal_cos = float(horizontal_cos)
        self.vertical_cos = float(vertical_cos)
        self.wall_min_inliers = int(wall_min_inliers)
        self.wall_protect_radius = float(wall_protect_radius)
        self.clutter_height_min = float(clutter_height_min)
        self.clutter_height_max = float(clutter_height_max)
        self.clutter_pad_factor = float(clutter_pad_factor)
        # fill
        self.fill_table_enabled = bool(fill_table)
        self.fill_grid_res = float(fill_grid_res)
        self.fill_occ_cell = float(fill_occ_cell)
        self.fill_occ_min_count = int(fill_occ_min_count)
        self.fill_close_iters = int(fill_close_iters)
        self.fill_dilate_iters = int(fill_dilate_iters)
        self.fill_near_existing = float(fill_near_existing)

    # ------------------------------------------------------ stage: clean

    def clean(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """pymeshlab cleanup cascade. Requires the ``mesh`` extra."""
        pml = _require("pymeshlab", "MeshRefiner.clean")

        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(
            vertex_matrix=verts.astype(np.float64),
            face_matrix=faces.astype(np.int32),
        ), "input")
        logger.info("clean: input %d verts / %d faces", verts.shape[0], faces.shape[0])

        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_t_vertices()
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_repair_non_manifold_vertices()

        # Floaters by absolute face count: catches thin slivers.
        ms.meshing_remove_connected_component_by_face_number(
            mincomponentsize=self.min_component_faces,
        )
        # Floaters by diameter (% of bbox diagonal).
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pml.PercentageValue(self.min_component_diag_pct),
        )
        ms.meshing_close_holes(maxholesize=self.max_hole_size)
        if self.laplacian_iters > 0:
            ms.apply_coord_laplacian_smoothing(stepsmoothnum=self.laplacian_iters)

        out = ms.current_mesh()
        nv = np.asarray(out.vertex_matrix(), dtype=np.float32)
        nf = np.asarray(out.face_matrix(),   dtype=np.int32)
        logger.info("clean: output %d verts / %d faces", nv.shape[0], nf.shape[0])
        return nv, nf

    # ------------------------------------------------------ stage: densify

    def densify(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Loop subdivision (×4 faces per iteration).

        Round-trips through a fresh ``MeshSet`` because leftover selection /
        topology flags from the cleanup pass otherwise suppress subdivision.
        """
        if self.subdivision_iters <= 0:
            return verts, faces
        pml = _require("pymeshlab", "MeshRefiner.densify")

        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(
            vertex_matrix=verts.astype(np.float64),
            face_matrix=faces.astype(np.int32),
        ))
        ms.meshing_surface_subdivision_loop(
            iterations=self.subdivision_iters,
            threshold=pml.PureValue(0.0),
        )
        out = ms.current_mesh()
        nv = np.asarray(out.vertex_matrix(), dtype=np.float32)
        nf = np.asarray(out.face_matrix(),   dtype=np.int32)
        logger.info("densify: %d iters → %d verts / %d faces",
                    self.subdivision_iters, nv.shape[0], nf.shape[0])
        return nv, nf

    # ------------------------------------------------------ stage: colorize

    def colorize(
        self,
        vertices: np.ndarray,
        normals: np.ndarray,
        label: str = "color",
    ) -> np.ndarray:
        """High-detail per-vertex colors from session RGB frames.

        Returns (N, 3) uint8. Caller decides whether to call :meth:`brighten`
        afterwards (or set ``brighten=True`` and let :meth:`refine` do it).
        """
        rgb_intr   = self._session.rgb_intrinsics
        depth_intr = self._session.depth_intrinsics
        n = len(vertices)
        if rgb_intr is None:
            return np.full((n, 3), 128, dtype=np.uint8)

        K_rgb = rgb_intr.K
        fx_r, fy_r, cx_r, cy_r = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]
        if depth_intr is not None:
            K_d = depth_intr.K
            fx_d, fy_d, cx_d, cy_d = K_d[0, 0], K_d[1, 1], K_d[0, 2], K_d[1, 2]
        R_l2o = self._session.R_optical_to_link.T

        color_sum  = np.zeros((n, 3), dtype=np.float64)
        weight_sum = np.zeros(n,       dtype=np.float64)
        n_occ = 0
        n_used = 0
        cap = self.color_max_frames

        for frame in tqdm(self._session.frames(),
                          total=self._session.num_rgb_frames,
                          desc=f"Coloring ({label})", unit="frame"):
            if cap is not None and n_used >= cap:
                break
            if frame.index % self.color_every_n != 0 or frame.camera_pose is None:
                continue
            n_used += 1
            R_w = frame.camera_pose.rotation
            t_w = frame.camera_pose.translation
            rgb = frame.rgb
            rh, rw = rgb.shape[:2]

            pts = (R_l2o @ (R_w.T @ (vertices - t_w).T)).T
            z = pts[:, 2]
            u_r = pts[:, 0] * fx_r / z + cx_r
            v_r = pts[:, 1] * fy_r / z + cy_r
            in_image = (z > 0.2) & (z < self.max_color_depth) \
                       & (u_r >= 0) & (u_r < rw - 1) & (v_r >= 0) & (v_r < rh - 1)
            if not np.any(in_image):
                continue

            view_dir = vertices - t_w
            view_dir /= (np.linalg.norm(view_dir, axis=1, keepdims=True) + 1e-9)
            cos_a = -(view_dir * normals).sum(axis=1)
            good = in_image & (cos_a > self.min_view_angle_cos)
            if not np.any(good):
                continue

            if self.color_use_occlusion and depth_intr is not None and frame.depth is not None:
                dh, dw = frame.depth.shape[:2]
                u_d = (pts[:, 0] * fx_d / z + cx_d).astype(np.int64)
                v_d = (pts[:, 1] * fy_d / z + cy_d).astype(np.int64)
                in_depth = (u_d >= 0) & (u_d < dw) & (v_d >= 0) & (v_d < dh)
                depth_at = np.zeros(n, dtype=np.float32)
                valid_d  = np.zeros(n, dtype=bool)
                ok = good & in_depth
                depth_at[ok] = frame.depth[v_d[ok], u_d[ok]].astype(np.float32) / 1000.0
                valid_d[ok]  = depth_at[ok] > 0.1
                # Pass occlusion if depth is missing (no negative evidence).
                visible = ~valid_d | (z <= depth_at + self.occlusion_tolerance)
                n_occ += int((good & ~visible).sum())
                good &= visible
                if not np.any(good):
                    continue

            ug = u_r[good]; vg = v_r[good]; zg = z[good]; cg = cos_a[good]
            col = _bilinear_sample(rgb, ug, vg)
            wt  = (cg ** 2) / (zg ** 2 + 1e-3)
            color_sum[good]  += col * wt[:, None]
            weight_sum[good] += wt

        out = np.full((n, 3), 128, dtype=np.uint8)
        hit = weight_sum > 0
        out[hit] = np.clip(color_sum[hit] / weight_sum[hit, None], 0, 255).astype(np.uint8)
        logger.info(
            "colorize: %d / %d verts coloured (%.1f%%) from %d frames "
            "(every_n=%d, occlusion=%s, %d rejects)",
            int(hit.sum()), n, 100 * hit.mean(), n_used,
            self.color_every_n, self.color_use_occlusion, n_occ,
        )
        return out

    def brighten_colors(self, colors: np.ndarray) -> np.ndarray:
        return brighten_colors(colors, self.brighten_factor, self.brighten_min)

    # ----------------------------------------- stage: strip table clutter

    def _multi_ransac(
        self, verts: np.ndarray, min_inliers: int,
    ) -> list[tuple[tuple, np.ndarray]]:
        """Iteratively fit and remove planes. Returns [(eq, orig_indices), ...]."""
        p3 = _require("pyransac3d", "MeshRefiner.strip_table_clutter")
        remaining = np.arange(len(verts))
        out = []
        for _ in range(self.ransac_max_planes):
            if len(remaining) < min_inliers:
                break
            rest = verts[remaining]
            eq, idx_in_rest = p3.Plane().fit(
                rest, thresh=self.ransac_dist_thresh, maxIteration=2000,
            )
            if len(idx_in_rest) < min_inliers:
                break
            orig_idx = remaining[idx_in_rest]
            out.append((tuple(eq), orig_idx))
            mask = np.ones(len(remaining), dtype=bool)
            mask[idx_in_rest] = False
            remaining = remaining[mask]
        return out

    def strip_table_clutter(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        colors: Optional[np.ndarray] = None,
    ) -> tuple[
        np.ndarray, np.ndarray, Optional[np.ndarray],
        Optional[tuple], Optional[np.ndarray],
    ]:
        """Detect the kitchen-counter plane and strip everything sitting on it.

        Walls attached to the counter are protected by kd-tree proximity to
        any RANSAC wall inlier.

        Returns ``(verts, faces, colors, table_plane_eq, table_inlier_points)``
        — the last two are returned for :meth:`fill_table`.
        """
        _require("scipy", "MeshRefiner.strip_table_clutter")
        _require("matplotlib", "MeshRefiner.strip_table_clutter")
        from scipy.spatial import ConvexHull, cKDTree
        from matplotlib.path import Path as MplPath

        n_total = len(verts)
        min_inl = max(200, int(n_total * self.ransac_min_inliers_pct / 100))
        planes_all = self._multi_ransac(verts, min_inl)
        logger.info("strip: %d planes detected (RANSAC)", len(planes_all))

        horiz = []
        walls_idx_sets = []
        for eq, orig_idx in planes_all:
            n = np.array(eq[:3], dtype=np.float64)
            n /= (np.linalg.norm(n) + 1e-9)
            cos_up = abs(float(n @ self.world_up))
            if cos_up > self.horizontal_cos:
                sample = -eq[3] * n
                h = float(sample @ self.world_up)
                horiz.append((eq, orig_idx, h))
                logger.info("strip:   horiz  y=%+.2f m  inliers=%d", h, len(orig_idx))
            elif cos_up < self.vertical_cos and len(orig_idx) >= self.wall_min_inliers:
                walls_idx_sets.append(orig_idx)
                logger.info("strip:   wall   |n·up|=%.2f  inliers=%d",
                            cos_up, len(orig_idx))

        if len(horiz) < 2:
            logger.warning("strip: only %d horizontal planes — cannot identify table",
                           len(horiz))
            return verts, faces, colors, None, None

        horiz.sort(key=lambda x: x[2])
        candidates = horiz[1:-1] if len(horiz) > 2 else horiz[1:]
        table = max(candidates, key=lambda x: len(x[1]))
        logger.info("strip: → selected table-top at y=%.2f m (%d inliers)",
                    table[2], len(table[1]))

        eq, table_idx, _ = table
        table_pts = verts[table_idx]
        p0, u, v, n = _plane_basis(eq)
        if n @ self.world_up < 0:
            n = -n

        uv_inl = np.column_stack([(table_pts - p0) @ u, (table_pts - p0) @ v])
        try:
            hull = ConvexHull(uv_inl)
        except Exception as e:
            logger.warning("strip: table hull failed (%s)", e)
            return verts, faces, colors, eq, table_pts
        boundary = uv_inl[hull.vertices]
        centroid = boundary.mean(0)
        boundary_pad = centroid + (boundary - centroid) * self.clutter_pad_factor

        rel = verts - p0
        uv_all = np.column_stack([rel @ u, rel @ v])
        h_above = rel @ n
        in_fp = MplPath(boundary_pad).contains_points(uv_all)
        is_clutter = in_fp & (h_above > self.clutter_height_min) \
                           & (h_above < self.clutter_height_max)

        wall_mask = np.zeros(n_total, dtype=bool)
        if walls_idx_sets:
            wall_pts = np.concatenate([verts[i] for i in walls_idx_sets], axis=0)
            d_to_wall, _ = cKDTree(wall_pts).query(verts, k=1)
            wall_mask = d_to_wall < self.wall_protect_radius
            logger.info("strip: wall-proximity protection: %d verts within %.0fcm",
                        int(wall_mask.sum()), self.wall_protect_radius * 100)
        n_saved = int((is_clutter & wall_mask).sum())
        is_clutter &= ~wall_mask
        logger.info("strip: wall protection rescued %d clutter-classified verts", n_saved)
        logger.info("strip: stripping %d / %d verts", int(is_clutter.sum()), n_total)

        keep_face = ~np.any(is_clutter[faces], axis=1)
        faces_kept = faces[keep_face]
        used = np.zeros(n_total, dtype=bool)
        used[faces_kept.flatten()] = True
        new_idx = np.full(n_total, -1, dtype=np.int64)
        new_idx[used] = np.arange(used.sum())
        verts_out = verts[used]
        faces_out = new_idx[faces_kept].astype(np.int32)
        colors_out = colors[used] if colors is not None else None
        logger.info("strip: result %d verts / %d faces",
                    verts_out.shape[0], faces_out.shape[0])
        return verts_out, faces_out, colors_out, eq, table_pts

    # ----------------------------------------- stage: fill table

    def fill_table(
        self,
        table_eq: tuple,
        table_inlier_pts: np.ndarray,
        kept_verts: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Flat patch on the counter plane. Region defined by an occupancy
        mask of inlier density (follows L-shaped counters; does not over-
        extend past the actual surface).

        Returns (verts, faces, normals) or None if no fill region.
        """
        _require("scipy", "MeshRefiner.fill_table")
        _require("matplotlib", "MeshRefiner.fill_table")
        from scipy.spatial import Delaunay, cKDTree
        from scipy.ndimage import binary_closing, binary_dilation
        from matplotlib.path import Path as MplPath  # noqa: F401  (parity)

        p0, u, v, n = _plane_basis(table_eq)
        if n @ self.world_up < 0:
            n = -n

        uv_inl = np.column_stack([(table_inlier_pts - p0) @ u,
                                  (table_inlier_pts - p0) @ v])
        if len(uv_inl) < 4:
            return None

        cell = self.fill_occ_cell
        mn = uv_inl.min(0) - cell * 5
        mx = uv_inl.max(0) + cell * 5
        nx = int(np.ceil((mx[0] - mn[0]) / cell))
        ny = int(np.ceil((mx[1] - mn[1]) / cell))
        ix = np.clip(((uv_inl[:, 0] - mn[0]) / cell).astype(int), 0, nx - 1)
        iy = np.clip(((uv_inl[:, 1] - mn[1]) / cell).astype(int), 0, ny - 1)
        occ = np.zeros((ny, nx), dtype=np.int32)
        np.add.at(occ, (iy, ix), 1)
        mask = occ >= self.fill_occ_min_count
        if self.fill_close_iters > 0:
            mask = binary_closing(mask, iterations=self.fill_close_iters)
        if self.fill_dilate_iters > 0:
            mask = binary_dilation(mask, iterations=self.fill_dilate_iters)
        logger.info("fill: occupancy mask %d / %d cells (%.0fx%.0f cm)",
                    int(mask.sum()), mask.size, cell * 100, cell * 100)

        res = self.fill_grid_res
        us = np.arange(mn[0], mx[0] + res, res)
        vs = np.arange(mn[1], mx[1] + res, res)
        UU, VV = np.meshgrid(us, vs)
        grid_uv = np.column_stack([UU.ravel(), VV.ravel()])
        gi = np.clip(((grid_uv[:, 0] - mn[0]) / cell).astype(int), 0, nx - 1)
        gj = np.clip(((grid_uv[:, 1] - mn[1]) / cell).astype(int), 0, ny - 1)
        grid_uv = grid_uv[mask[gj, gi]]
        if len(grid_uv) < 3:
            return None

        grid_xyz = (p0 + grid_uv[:, 0:1] * u + grid_uv[:, 1:2] * v).astype(np.float32)
        d, _ = cKDTree(kept_verts).query(grid_xyz, k=1)
        far = d > self.fill_near_existing
        if far.sum() < 3:
            return None
        grid_xyz = grid_xyz[far]
        grid_uv  = grid_uv[far]

        try:
            tri = Delaunay(grid_uv)
        except Exception:
            return None
        max_edge = res * 2.5
        s = tri.simplices
        e0 = np.linalg.norm(grid_uv[s[:, 1]] - grid_uv[s[:, 0]], axis=1)
        e1 = np.linalg.norm(grid_uv[s[:, 2]] - grid_uv[s[:, 1]], axis=1)
        e2 = np.linalg.norm(grid_uv[s[:, 0]] - grid_uv[s[:, 2]], axis=1)
        keep_tri = (e0 < max_edge) & (e1 < max_edge) & (e2 < max_edge)
        faces_p = s[keep_tri].astype(np.int32)
        if len(faces_p) < 1:
            return None

        normals_p = np.tile(n.astype(np.float32), (len(grid_xyz), 1))
        logger.info("fill: patch %d verts / %d faces", len(grid_xyz), len(faces_p))
        return grid_xyz, faces_p, normals_p

    # ----------------------------------------- one-shot

    def refine(
        self,
        verts: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
    ) -> RefinedMesh:
        """Run the full pipeline. If ``verts`` and ``faces`` are not given,
        loads the mesh from ``session.mesh()``."""
        if verts is None or faces is None:
            mesh_data = self._session.mesh()
            if mesh_data is None:
                raise RuntimeError("session has no /map/mesh topic; cannot refine")
            verts, faces, _ = mesh_data
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)

        if self.cleanup:
            verts, faces = self.clean(verts, faces)
        if self.subdivision_iters > 0:
            verts, faces = self.densify(verts, faces)

        normals = compute_vertex_normals(verts, faces)
        colors: Optional[np.ndarray] = None
        if self.colorize_enabled:
            colors = self.colorize(verts, normals, label="refine")
            if self.brighten:
                colors = self.brighten_colors(colors)

        if self.strip_table_clutter_enabled:
            verts, faces, colors, table_eq, table_pts = self.strip_table_clutter(
                verts, faces, colors,
            )
            if self.fill_table_enabled and table_eq is not None:
                patch = self.fill_table(table_eq, table_pts, kept_verts=verts)
                if patch is not None:
                    fv, ff, fn = patch
                    fill_colors = None
                    if self.colorize_enabled:
                        fill_colors = self.colorize(fv, fn, label="fill")
                        if self.brighten:
                            fill_colors = self.brighten_colors(fill_colors)
                    offset = len(verts)
                    verts = np.concatenate([verts, fv], axis=0).astype(np.float32)
                    faces = np.concatenate([faces, ff + offset], axis=0).astype(np.int32)
                    if colors is not None and fill_colors is not None:
                        colors = np.concatenate([colors, fill_colors], axis=0).astype(np.uint8)
                    logger.info("refine: merged fill → %d verts / %d faces",
                                verts.shape[0], faces.shape[0])
            # geometry changed → recompute normals
            normals = compute_vertex_normals(verts, faces)

        return RefinedMesh(
            vertices=verts,
            faces=faces,
            vertex_colors=colors,
            vertex_normals=normals,
        )
