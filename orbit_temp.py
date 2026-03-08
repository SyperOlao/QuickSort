import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from dataclasses import dataclass


# =========================================================
# МАТЕМАТИКА
# =========================================================
def solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 30) -> float:
    """
    Решает уравнение Кеплера:
        M = E - e*sin(E)
    """
    M = np.mod(M, 2 * np.pi)

    E = M if e < 0.8 else np.pi

    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E += dE
        if abs(dE) < tol:
            break

    return E


def kepler_local_position(a: float, e: float, inc_deg: float, M: float) -> np.ndarray:
    """
    Позиция точки на эллиптической орбите в локальной системе координат.
    Орбита лежит в XY, потом наклоняется вокруг оси X.
    """
    E = solve_kepler(M, e)

    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e**2) * np.sin(E)

    inc = np.radians(inc_deg)

    x = x_orb
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)

    return np.array([x, y, z], dtype=float)


def kepler_local_curve(a: float, e: float, inc_deg: float, num: int = 400):
    """
    Точки орбиты для отрисовки линии.
    """
    E_vals = np.linspace(0, 2 * np.pi, num)

    x_orb = a * (np.cos(E_vals) - e)
    y_orb = a * np.sqrt(1 - e**2) * np.sin(E_vals)

    inc = np.radians(inc_deg)

    x = x_orb
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)

    return x, y, z


# =========================================================
# СТРУКТУРЫ ДАННЫХ
# =========================================================
@dataclass
class OrbitParams:
    a: float                  # большая полуось
    e: float                  # эксцентриситет
    inc_deg: float            # наклон
    angular_speed: float      # скорость изменения средней аномалии
    phase: float = 0.0        # фазовый сдвиг


class Anchor:
    """
    Неподвижная точка. Например, корневой барицентр.
    """
    def __init__(self, name: str, position=(0.0, 0.0, 0.0)):
        self.name = name
        self.position = np.array(position, dtype=float)

    def get_world_position(self) -> np.ndarray:
        return self.position


class OrbitalBody:
    """
    Универсальный объект:
    - звезда
    - планета
    - луна
    Всё это просто тело на орбите вокруг parent.
    """
    def __init__(
        self,
        name: str,
        parent,
        orbit: OrbitParams,
        color: str = "b",
        marker: str = "o",
        size: int = 6,
        orbit_style: str = "--",
        orbit_alpha: float = 0.8,
        draw_orbit: bool = True
    ):
        self.name = name
        self.parent = parent
        self.orbit = orbit
        self.color = color
        self.marker = marker
        self.size = size
        self.orbit_style = orbit_style
        self.orbit_alpha = orbit_alpha
        self.draw_orbit = draw_orbit

        self.mean_anomaly = orbit.phase

        self.point_artist = None
        self.orbit_artist = None

    def update(self, dt: float):
        self.mean_anomaly += self.orbit.angular_speed * dt

    def get_local_position(self) -> np.ndarray:
        return kepler_local_position(
            self.orbit.a,
            self.orbit.e,
            self.orbit.inc_deg,
            self.mean_anomaly
        )

    def get_world_position(self) -> np.ndarray:
        parent_pos = self.parent.get_world_position()
        return parent_pos + self.get_local_position()

    def get_world_orbit_curve(self):
        px, py, pz = self.parent.get_world_position()
        x, y, z = kepler_local_curve(
            self.orbit.a,
            self.orbit.e,
            self.orbit.inc_deg
        )
        return x + px, y + py, z + pz

    def create_artists(self, ax):
        self.point_artist, = ax.plot(
            [], [], [],
            color=self.color,
            marker=self.marker,
            linestyle="None",
            markersize=self.size,
            label=self.name
        )

        if self.draw_orbit:
            self.orbit_artist, = ax.plot(
                [], [], [],
                color=self.color,
                linestyle=self.orbit_style,
                linewidth=1,
                alpha=self.orbit_alpha
            )

    def refresh_artists(self):
        pos = self.get_world_position()

        self.point_artist.set_data([pos[0]], [pos[1]])
        self.point_artist.set_3d_properties([pos[2]])

        if self.draw_orbit and self.orbit_artist is not None:
            x, y, z = self.get_world_orbit_curve()
            self.orbit_artist.set_data(x, y)
            self.orbit_artist.set_3d_properties(z)


# =========================================================
# СЦЕНА
# =========================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.22)

ax.set_xlim(-60, 60)
ax.set_ylim(-60, 60)
ax.set_zlim(-60, 60)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Refactored Orbital System")

ax.view_init(elev=25, azim=35)

try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    pass


# =========================================================
# КОРНЕВОЙ БАРИЦЕНТР
# =========================================================
root_barycenter = Anchor("RootBarycenter", (0.0, 0.0, 0.0))

# рисуем сам барицентр
bary_artist, = ax.plot([0], [0], [0], "r+", markersize=12)


# =========================================================
# ПАРАМЕТРЫ
# =========================================================
A_BIN = 16.0
E_BIN = 0.4
INCL_BIN = 20.0

A_PLANET_1 = 35.0
E_PLANET_1 = 0.15

A_MOON_1 = 4.0
E_MOON_1 = 0.05
INCL_MOON_1 = 30.0

# Базовые скорости
BIN_SPEED = 0.045
PLANET_SPEED = 0.012
MOON_SPEED = 0.08


# =========================================================
# ОБЪЕКТЫ
# =========================================================
bodies = []

# две звезды вокруг барицентра
star1 = OrbitalBody(
    name="Star 1",
    parent=root_barycenter,
    orbit=OrbitParams(a=A_BIN / 2, e=E_BIN, inc_deg=INCL_BIN, angular_speed=BIN_SPEED, phase=0.0),
    color="gold",
    marker="o",
    size=10
)

star2 = OrbitalBody(
    name="Star 2",
    parent=root_barycenter,
    orbit=OrbitParams(a=A_BIN / 2, e=E_BIN, inc_deg=INCL_BIN, angular_speed=BIN_SPEED, phase=np.pi),
    color="orange",
    marker="o",
    size=10
)

# планета вокруг барицентра
planet1 = OrbitalBody(
    name="Planet 1",
    parent=root_barycenter,
    orbit=OrbitParams(a=A_PLANET_1, e=E_PLANET_1, inc_deg=INCL_BIN, angular_speed=PLANET_SPEED, phase=0.3),
    color="deepskyblue",
    marker="o",
    size=6
)

# луна вокруг планеты
moon1 = OrbitalBody(
    name="Moon 1",
    parent=planet1,
    orbit=OrbitParams(a=A_MOON_1, e=E_MOON_1, inc_deg=INCL_MOON_1, angular_speed=MOON_SPEED, phase=0.0),
    color="lime",
    marker="o",
    size=4
)

bodies.extend([star1, star2, planet1, moon1])


# =========================================================
# ХОЧЕШЬ ДОБАВИТЬ НОВУЮ ПЛАНЕТУ? ВОТ ТАК:
# =========================================================
planet2 = OrbitalBody(
    name="Planet 2",
    parent=root_barycenter,
    orbit=OrbitParams(a=48.0, e=0.08, inc_deg=10.0, angular_speed=0.008, phase=1.5),
    color="violet",
    marker="o",
    size=5
)

moon2 = OrbitalBody(
    name="Moon 2",
    parent=planet2,
    orbit=OrbitParams(a=3.0, e=0.0, inc_deg=15.0, angular_speed=0.11, phase=0.7),
    color="red",
    marker="o",
    size=3
)

bodies.extend([planet2, moon2])


# =========================================================
# СОЗДАНИЕ ГРАФИЧЕСКИХ ОБЪЕКТОВ
# =========================================================
for body in bodies:
    body.create_artists(ax)


# =========================================================
# СЛАЙДЕРЫ
# =========================================================
slider_left = 0.18
slider_width = 0.67
slider_height = 0.03
slider_gap = 0.06
start_y = 0.12

plt.subplots_adjust(bottom=0.25)

ax_speed    = fig.add_axes((slider_left, start_y,                  slider_width, slider_height))
ax_planet_a = fig.add_axes((slider_left, start_y - slider_gap,     slider_width, slider_height))
ax_moon_a   = fig.add_axes((slider_left, start_y - 2 * slider_gap, slider_width, slider_height))

slider_speed = Slider(ax_speed, "speed", 0.1, 5.0, valinit=1.0)
slider_planet_a = Slider(ax_planet_a, "planet orbit", 20.0, 80.0, valinit=A_PLANET_1)
slider_moon_a = Slider(ax_moon_a, "moon orbit", 1.0, 20.0, valinit=A_MOON_1)


# =========================================================
# ИНИЦИАЛИЗАЦИЯ
# =========================================================
def init():
    for body in bodies:
        body.refresh_artists()

    artists = [bary_artist]
    for body in bodies:
        artists.append(body.point_artist)
        if body.orbit_artist is not None:
            artists.append(body.orbit_artist)

    return artists


# =========================================================
# ОБНОВЛЕНИЕ КАДРА
# =========================================================
def update(frame):
    speed = slider_speed.val

    # можно менять полуось первой планеты слайдером
    planet1.orbit.a = slider_planet_a.val
    moon1.orbit.a = slider_moon_a.val
    moon2.orbit.a = slider_planet_a.val
    # dt здесь по сути "масштаб времени", а не физические секунды
    dt = speed

    for body in bodies:
        body.update(dt)

    for body in bodies:
        body.refresh_artists()

    artists = [bary_artist]
    for body in bodies:
        artists.append(body.point_artist)
        if body.orbit_artist is not None:
            artists.append(body.orbit_artist)

    return artists


ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=40,
    blit=False,
    cache_frame_data=False
)

plt.show()