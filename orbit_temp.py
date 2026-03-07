import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


# -----------------------------
# ПАРАМЕТРЫ СИСТЕМЫ
# -----------------------------
A_BIN = 16.0         # большая полуось относительной орбиты звезд
E_BIN = 0.4          # эксцентриситет орбиты звезд
INCL_DEG = 20.0      # наклон орбитальной плоскости

A_PLANET_INIT = 35.0
E_PLANET = 0.15

# базовые "скорости времени"
N_BIN_BASE = 0.045
N_PLANET_BASE = 0.012

# глобальное время
t_bin = 0.0
t_planet = 0.0


# -----------------------------
# МАТЕМАТИКА
# -----------------------------
def solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 30) -> float:
    """
    Решение уравнения Кеплера:
        M = E - e*sin(E)

    M - средняя аномалия
    E - эксцентрическая аномалия
    """
    M = np.mod(M, 2 * np.pi)

    if e < 0.8:
        E = M
    else:
        E = np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE

        if abs(dE) < tol:
            break

    return E


def kepler_position(a: float, e: float, inc_deg: float, M: float) -> tuple[float, float, float]:
    """
    Координаты точки на эллиптической орбите через среднюю аномалию.
    Сначала орбита в плоскости XY, затем наклон вокруг оси X.
    """
    E = solve_kepler(M, e)

    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e**2) * np.sin(E)

    inc = np.radians(inc_deg)

    x = x_orb
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)

    return x, y, z


def build_orbit_curve(a: float, e: float, inc_deg: float, num: int = 500):
    """
    Строит точки всей орбиты для отображения линии.
    """
    E_vals = np.linspace(0, 2 * np.pi, num)

    x_orb = a * (np.cos(E_vals) - e)
    y_orb = a * np.sqrt(1 - e**2) * np.sin(E_vals)

    inc = np.radians(inc_deg)

    x = x_orb
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)

    return x, y, z


# -----------------------------
# ФИГУРА
# -----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.22)

ax.set_xlim(-50, 50)
ax.set_ylim(-50, 50)
ax.set_zlim(-50, 50)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Circumbinary Orbit Simulation")

ax.view_init(elev=25, azim=35)

# фикс масштаба, чтобы 3D не косило сцену слишком мерзко
try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    pass


# -----------------------------
# ОБЪЕКТЫ СЦЕНЫ
# -----------------------------
# звезды
star1, = ax.plot([], [], [], "yo", markersize=10)
star2, = ax.plot([], [], [], "yo", markersize=10)

# планета
planet, = ax.plot([], [], [], "bo", markersize=6)

# барицентр
barycenter, = ax.plot([0], [0], [0], "r+", markersize=12)

# орбиты
orbit_star1, = ax.plot([], [], [], "y--", linewidth=1, alpha=0.8)
orbit_star2, = ax.plot([], [], [], "y--", linewidth=1, alpha=0.8)
orbit_planet, = ax.plot([], [], [], "b--", linewidth=1, alpha=0.8)


# -----------------------------
# СЛАЙДЕРЫ
# -----------------------------
ax_speed = plt.axes((0.18, 0.10, 0.67, 0.03))
ax_planet_a = plt.axes((0.18, 0.05, 0.67, 0.03))

slider_speed = Slider(ax_speed, "speed", 0.1, 5.0, valinit=1.0)
slider_planet_a = Slider(ax_planet_a, "planet orbit", 25.0, 80.0, valinit=A_PLANET_INIT)


# -----------------------------
# ИНИЦИАЛИЗАЦИЯ
# -----------------------------
def init():
    # каждая звезда, при равных массах, ходит по полуоси A_BIN/2
    xs, ys, zs = build_orbit_curve(A_BIN / 2.0, E_BIN, INCL_DEG)
    orbit_star1.set_data(xs, ys)
    orbit_star1.set_3d_properties(zs)

    orbit_star2.set_data(-xs, -ys)
    orbit_star2.set_3d_properties(-zs)

    xp, yp, zp = build_orbit_curve(A_PLANET_INIT, E_PLANET, INCL_DEG)
    orbit_planet.set_data(xp, yp)
    orbit_planet.set_3d_properties(zp)

    return (
        star1, star2, planet, barycenter,
        orbit_star1, orbit_star2, orbit_planet
    )


# -----------------------------
# ОБНОВЛЕНИЕ КАДРА
# -----------------------------
def update(frame):
    global t_bin, t_planet

    speed = slider_speed.val
    planet_a = slider_planet_a.val

    # обновляем внутреннее "время"
    t_bin += N_BIN_BASE * speed
    t_planet += N_PLANET_BASE * speed

    # звезды вокруг барицентра
    x1, y1, z1 = kepler_position(A_BIN / 2.0, E_BIN, INCL_DEG, t_bin)
    x2, y2, z2 = -x1, -y1, -z1

    star1.set_data([x1], [y1])
    star1.set_3d_properties([z1])

    star2.set_data([x2], [y2])
    star2.set_3d_properties([z2])

    # планета вокруг барицентра
    px, py, pz = kepler_position(planet_a, E_PLANET, INCL_DEG, t_planet)
    planet.set_data([px], [py])
    planet.set_3d_properties([pz])

    # если двигают слайдер орбиты планеты, перестраиваем линию
    xp, yp, zp = build_orbit_curve(planet_a, E_PLANET, INCL_DEG)
    orbit_planet.set_data(xp, yp)
    orbit_planet.set_3d_properties(zp)

    return (
        star1, star2, planet, barycenter,
        orbit_star1, orbit_star2, orbit_planet
    )


ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=40,
    blit=False,
    cache_frame_data=False
)

plt.show()