from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pyvista as pv


# =========================
# Math / Noise
# =========================

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v.copy()
    return v / norm


def fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def hash3(ix: int, iy: int, iz: int, seed: int) -> float:
    n = ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 1274126177
    n = (n ^ (n >> 13)) * 1274126177
    n = n ^ (n >> 16)
    return (n & 0xFFFFFFFF) / 0xFFFFFFFF


def random_gradient(ix: int, iy: int, iz: int, seed: int) -> np.ndarray:
    u = hash3(ix, iy, iz, seed)
    v = hash3(ix + 17, iy - 31, iz + 47, seed + 101)

    theta = 2.0 * math.pi * u
    z = 2.0 * v - 1.0
    r = math.sqrt(max(0.0, 1.0 - z * z))

    return np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=np.float64)


def gradient_noise_3d(x: float, y: float, z: float, seed: int) -> float:
    x0 = math.floor(x)
    y0 = math.floor(y)
    z0 = math.floor(z)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    sx = fade(x - x0)
    sy = fade(y - y0)
    sz = fade(z - z0)

    def dot_grid(ix: int, iy: int, iz: int) -> float:
        grad = random_gradient(ix, iy, iz, seed)
        dx = x - ix
        dy = y - iy
        dz = z - iz
        return grad[0] * dx + grad[1] * dy + grad[2] * dz

    n000 = dot_grid(x0, y0, z0)
    n100 = dot_grid(x1, y0, z0)
    n010 = dot_grid(x0, y1, z0)
    n110 = dot_grid(x1, y1, z0)
    n001 = dot_grid(x0, y0, z1)
    n101 = dot_grid(x1, y0, z1)
    n011 = dot_grid(x0, y1, z1)
    n111 = dot_grid(x1, y1, z1)

    nx00 = lerp(n000, n100, sx)
    nx10 = lerp(n010, n110, sx)
    nx01 = lerp(n001, n101, sx)
    nx11 = lerp(n011, n111, sx)

    nxy0 = lerp(nx00, nx10, sy)
    nxy1 = lerp(nx01, nx11, sy)

    return lerp(nxy0, nxy1, sz)


def fbm_noise_3d(
    p: np.ndarray,
    seed: int,
    octaves: int,
    frequency: float,
    lacunarity: float,
    gain: float,
) -> float:
    value = 0.0
    amplitude = 1.0
    freq = frequency
    amplitude_sum = 0.0

    for i in range(octaves):
        n = gradient_noise_3d(p[0] * freq, p[1] * freq, p[2] * freq, seed + i * 97)
        value += n * amplitude
        amplitude_sum += amplitude
        amplitude *= gain
        freq *= lacunarity

    return value / amplitude_sum if amplitude_sum > 1e-12 else 0.0


def ridged_fbm_noise_3d(
    p: np.ndarray,
    seed: int,
    octaves: int,
    frequency: float,
    lacunarity: float,
    gain: float,
) -> float:
    value = 0.0
    amplitude = 1.0
    freq = frequency
    amplitude_sum = 0.0

    for i in range(octaves):
        n = gradient_noise_3d(p[0] * freq, p[1] * freq, p[2] * freq, seed + i * 131)
        n = 1.0 - abs(n)
        n = n * n
        value += n * amplitude
        amplitude_sum += amplitude
        amplitude *= gain
        freq *= lacunarity

    return value / amplitude_sum if amplitude_sum > 1e-12 else 0.0


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-12:
        return 0.0 if x < edge0 else 1.0
    t = (x - edge0) / (edge1 - edge0)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


# =========================
# Settings
# =========================

@dataclass
class PlanetNoiseSettings:
    radius: float = 1.0
    resolution: int = 96

    continent_frequency: float = 1.5
    continent_octaves: int = 5
    continent_gain: float = 0.5
    continent_lacunarity: float = 2.0
    continent_strength: float = 0.20

    mountain_frequency: float = 10.0
    mountain_octaves: int = 6
    mountain_gain: float = 0.5
    mountain_lacunarity: float = 2.1
    mountain_strength: float = 0.08

    mountain_mask_start: float = 0.05
    mountain_mask_end: float = 0.25

    seed: int = 42


@dataclass
class WaterSettings:
    water_level_01: float = 0.42
    sea_epsilon: float = 1e-5


FACE_NORMALS = [
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
    np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, 1.0], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
]


# =========================
# Planet surface
# =========================

def face_axes(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if abs(normal[0]) > 0.5:
        axis_a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif abs(normal[1]) > 0.5:
        axis_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis_a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        axis_b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return axis_a, axis_b


def evaluate_planet_height(direction: np.ndarray, settings: PlanetNoiseSettings) -> float:
    continent = fbm_noise_3d(
        direction,
        seed=settings.seed,
        octaves=settings.continent_octaves,
        frequency=settings.continent_frequency,
        lacunarity=settings.continent_lacunarity,
        gain=settings.continent_gain,
    )
    continent01 = continent * 0.5 + 0.5
    base_height = (continent01 - 0.5) * 2.0 * settings.continent_strength

    mountain = ridged_fbm_noise_3d(
        direction,
        seed=settings.seed + 10000,
        octaves=settings.mountain_octaves,
        frequency=settings.mountain_frequency,
        lacunarity=settings.mountain_lacunarity,
        gain=settings.mountain_gain,
    )

    mountain_mask = smoothstep(
        settings.mountain_mask_start,
        settings.mountain_mask_end,
        continent01
    )

    mountain_height = mountain * settings.mountain_strength * mountain_mask
    return base_height + mountain_height


def generate_cube_sphere_planet(
    settings: PlanetNoiseSettings
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    returns:
        vertices: (N,3)
        triangles: (M,3)
        heights: (N,)
        radii: (N,)
    """
    resolution = settings.resolution
    if resolution < 2:
        raise ValueError("resolution must be >= 2")

    vertices: List[np.ndarray] = []
    heights: List[float] = []
    radii: List[float] = []
    triangles: List[Tuple[int, int, int]] = []

    for face_normal in FACE_NORMALS:
        axis_a, axis_b = face_axes(face_normal)
        face_indices = np.zeros((resolution, resolution), dtype=np.int32)

        for y in range(resolution):
            for x in range(resolution):
                u = x / (resolution - 1)
                v = y / (resolution - 1)

                local_x = (u - 0.5) * 2.0
                local_y = (v - 0.5) * 2.0

                point_on_cube = face_normal + local_x * axis_a + local_y * axis_b
                direction = normalize(point_on_cube)

                height = evaluate_planet_height(direction, settings)
                final_radius = settings.radius + height
                position = direction * final_radius

                idx = len(vertices)
                vertices.append(position)
                heights.append(height)
                radii.append(final_radius)
                face_indices[y, x] = idx

        for y in range(resolution - 1):
            for x in range(resolution - 1):
                i0 = int(face_indices[y, x])
                i1 = int(face_indices[y, x + 1])
                i2 = int(face_indices[y + 1, x])
                i3 = int(face_indices[y + 1, x + 1])

                triangles.append((i0, i3, i2))
                triangles.append((i0, i1, i3))

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(heights, dtype=np.float64),
        np.asarray(radii, dtype=np.float64),
    )


# =========================
# Water / Sea clipping
# =========================

@dataclass
class LiquidSample:
    direction: np.ndarray
    terrain_radius: float
    underwater: bool


def compute_sea_radius(radii: np.ndarray, water_settings: WaterSettings) -> float:
    min_r = float(np.min(radii))
    max_r = float(np.max(radii))
    t = max(0.0, min(1.0, water_settings.water_level_01))
    return lerp(min_r, max_r, t)


def make_liquid_vertex(sample: LiquidSample, sea_radius: float) -> np.ndarray:
    return normalize(sample.direction) * sea_radius


def intersect_sea_edge(a: LiquidSample, b: LiquidSample, sea_radius: float) -> np.ndarray:
    da = a.terrain_radius - sea_radius
    db = b.terrain_radius - sea_radius
    denom = da - db

    t = 0.5
    if abs(denom) > 1e-12:
        t = da / denom
    t = max(0.0, min(1.0, t))

    dir_interp = normalize(a.direction * (1.0 - t) + b.direction * t)
    return dir_interp * sea_radius


def append_triangle(
    vertices_out: List[np.ndarray],
    triangles_out: List[Tuple[int, int, int]],
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> None:
    i0 = len(vertices_out)
    vertices_out.append(p0)
    i1 = len(vertices_out)
    vertices_out.append(p1)
    i2 = len(vertices_out)
    vertices_out.append(p2)
    triangles_out.append((i0, i1, i2))


def append_clipped_liquid_triangle(
    vertices_out: List[np.ndarray],
    triangles_out: List[Tuple[int, int, int]],
    a: LiquidSample,
    b: LiquidSample,
    c: LiquidSample,
    sea_radius: float,
) -> None:
    ba = a.underwater
    bb = b.underwater
    bc = c.underwater

    under_count = int(ba) + int(bb) + int(bc)

    if under_count == 0:
        return

    if under_count == 3:
        append_triangle(
            vertices_out,
            triangles_out,
            make_liquid_vertex(a, sea_radius),
            make_liquid_vertex(b, sea_radius),
            make_liquid_vertex(c, sea_radius),
        )
        return

    if under_count == 1:
        if ba:
            inside = a
            out1 = b
            out2 = c
        elif bb:
            inside = b
            out1 = c
            out2 = a
        else:
            inside = c
            out1 = a
            out2 = b

        p0 = make_liquid_vertex(inside, sea_radius)
        p1 = intersect_sea_edge(inside, out1, sea_radius)
        p2 = intersect_sea_edge(inside, out2, sea_radius)

        append_triangle(vertices_out, triangles_out, p0, p1, p2)
        return

    if under_count == 2:
        if not ba:
            outside = a
            in1 = b
            in2 = c
        elif not bb:
            outside = b
            in1 = c
            in2 = a
        else:
            outside = c
            in1 = a
            in2 = b

        p0 = make_liquid_vertex(in1, sea_radius)
        p1 = make_liquid_vertex(in2, sea_radius)
        p2 = intersect_sea_edge(in2, outside, sea_radius)
        p3 = intersect_sea_edge(in1, outside, sea_radius)

        append_triangle(vertices_out, triangles_out, p0, p1, p2)
        append_triangle(vertices_out, triangles_out, p0, p2, p3)
        return


def build_water_mesh_from_planet(
    vertices: np.ndarray,
    triangles: np.ndarray,
    radii: np.ndarray,
    sea_radius: float,
    water_settings: WaterSettings,
) -> Tuple[np.ndarray, np.ndarray]:
    water_vertices: List[np.ndarray] = []
    water_triangles: List[Tuple[int, int, int]] = []

    sea_eps = water_settings.sea_epsilon

    for tri in triangles:
        i0, i1, i2 = map(int, tri)

        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]

        r0 = radii[i0]
        r1 = radii[i1]
        r2 = radii[i2]

        s0 = LiquidSample(
            direction=normalize(p0),
            terrain_radius=float(r0),
            underwater=bool(r0 <= sea_radius + sea_eps),
        )
        s1 = LiquidSample(
            direction=normalize(p1),
            terrain_radius=float(r1),
            underwater=bool(r1 <= sea_radius + sea_eps),
        )
        s2 = LiquidSample(
            direction=normalize(p2),
            terrain_radius=float(r2),
            underwater=bool(r2 <= sea_radius + sea_eps),
        )

        append_clipped_liquid_triangle(
            water_vertices,
            water_triangles,
            s0, s1, s2,
            sea_radius
        )

    if not water_vertices:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
        )

    return (
        np.asarray(water_vertices, dtype=np.float64),
        np.asarray(water_triangles, dtype=np.int32),
    )


# =========================
# Mesh utils
# =========================

def build_pyvista_mesh(vertices: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    if len(vertices) == 0 or len(triangles) == 0:
        return pv.PolyData()

    faces = np.hstack([
        np.full((triangles.shape[0], 1), 3, dtype=np.int32),
        triangles
    ]).ravel()

    mesh = pv.PolyData(vertices, faces)
    mesh.compute_normals(inplace=True)
    return mesh


# =========================
# Rendering
# =========================

def show_planet_with_water(
    land_vertices: np.ndarray,
    land_triangles: np.ndarray,
    heights: np.ndarray,
    water_vertices: np.ndarray,
    water_triangles: np.ndarray,
    sea_radius: float,
) -> None:
    land_mesh = build_pyvista_mesh(land_vertices, land_triangles)
    land_mesh["height"] = heights

    water_mesh = build_pyvista_mesh(water_vertices, water_triangles)

    plotter = pv.Plotter(window_size=(1600, 950))
    plotter.set_background("#0b1020")

    # Суша
    plotter.add_mesh(
        land_mesh,
        scalars="height",
        cmap="terrain",
        smooth_shading=True,
        show_edges=False,
        specular=0.08,
    )

    # Вода
    if water_mesh.n_points > 0:
        plotter.add_mesh(
            water_mesh,
            color="#3b82f6",
            smooth_shading=True,
            show_edges=False,
            opacity=0.92,
            specular=0.35,
        )

    light1 = pv.Light(position=(5, 3, 2), focal_point=(0, 0, 0), intensity=1.2)
    light2 = pv.Light(position=(-4, -2, 1), focal_point=(0, 0, 0), intensity=0.35)

    plotter.add_light(light1)
    plotter.add_light(light2)

    plotter.add_axes()
    plotter.show_grid()
    plotter.add_text(
        f"Procedural Planet | Sea Radius = {sea_radius:.4f}",
        position="upper_left",
        font_size=12
    )

    plotter.camera_position = "iso"
    plotter.show()


# =========================
# Main
# =========================

if __name__ == "__main__":
    planet_settings = PlanetNoiseSettings(
        radius=1.0,
        resolution=128,
        continent_frequency=1.5,
        continent_octaves=5,
        continent_strength=0.20,
        mountain_frequency=10.0,
        mountain_octaves=6,
        mountain_strength=0.08,
        seed=42,
    )

    water_settings = WaterSettings(
        water_level_01=0.42,   # ниже = меньше воды, выше = больше
        sea_epsilon=1e-5,
    )

    land_vertices, land_triangles, heights, radii = generate_cube_sphere_planet(planet_settings)

    sea_radius = compute_sea_radius(radii, water_settings)

    water_vertices, water_triangles = build_water_mesh_from_planet(
        land_vertices,
        land_triangles,
        radii,
        sea_radius,
        water_settings,
    )

    print("Land vertices:", land_vertices.shape)
    print("Land triangles:", land_triangles.shape)
    print("Water vertices:", water_vertices.shape)
    print("Water triangles:", water_triangles.shape)
    print("Sea radius:", sea_radius)

    show_planet_with_water(
        land_vertices,
        land_triangles,
        heights,
        water_vertices,
        water_triangles,
        sea_radius,
    )