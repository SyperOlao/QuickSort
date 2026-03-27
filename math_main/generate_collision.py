import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
import trimesh
from vhacdx import compute_vhacd
import pyglet

def generate_torus(R=3.0, r=1.0, nu=64, nv=32, noise_amp=0.15):
    u = np.linspace(0, 2*np.pi, nu, endpoint=False)
    v = np.linspace(0, 2*np.pi, nv, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing='ij')

    # базовая поверхность тора
    cx = np.cos(uu)
    sx = np.sin(uu)
    cv = np.cos(vv)
    sv = np.sin(vv)

    x = (R + r * cv) * cx
    y = (R + r * cv) * sx
    z = r * sv

    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # локальная "наружная" нормаль тора
    nx = cv * cx
    ny = cv * sx
    nz = sv
    normals = np.stack([nx, ny, nz], axis=-1).reshape(-1, 3)

    # простой шум, потом можно заменить на perlin/simplex
    p = vertices.copy()
    noise = (
        0.6 * np.sin(2.3 * p[:, 0] + 1.1 * p[:, 1]) +
        0.3 * np.sin(3.7 * p[:, 1] - 0.9 * p[:, 2]) +
        0.2 * np.sin(5.1 * p[:, 2] + 0.4 * p[:, 0])
    )
    noise /= np.max(np.abs(noise))

    vertices = vertices + normals * (noise_amp * noise[:, None])

    def idx(i, j):
        return i * nv + j

    faces = []
    for i in range(nu):
        for j in range(nv):
            i2 = (i + 1) % nu
            j2 = (j + 1) % nv

            a = idx(i, j)
            b = idx(i2, j)
            c = idx(i2, j2)
            d = idx(i, j2)

            faces.append([a, b, c])
            faces.append([a, c, d])

    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    return mesh

mesh = generate_torus()

parts = mesh.convex_decomposition(
    maxConvexHulls=2000,
    resolution=100000,
    minimumVolumePercentErrorAllowed=1.0,
    maxRecursionDepth=10,
    shrinkWrap=True
)


scene = trimesh.Scene()

for part in parts:
    scene.add_geometry(part)

scene.show()
# plt.show()