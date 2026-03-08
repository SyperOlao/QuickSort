import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider


# =========================================================
# ТИПЫ
# =========================================================
BodyType = Literal["BH", "Star", "Planet", "Moon"]


# =========================================================
# RANGE
# =========================================================
@dataclass
class RangeF:
    min: float
    max: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.min, self.max)


@dataclass
class RangeI:
    min: int
    max: int

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.min, self.max)


# =========================================================
# ПРЕСЕТЫ
# =========================================================
@dataclass
class BodyPreset:
    body_type: BodyType
    mass: RangeF
    radius: RangeF
    eccentricity: RangeF
    inclination_deg: RangeF
    orbit_distance: RangeF
    orbit_speed: RangeF
    moon_count: RangeI = field(default_factory=lambda: RangeI(0, 0))


# =========================================================
# ДАННЫЕ ОРБИТЫ
# =========================================================
@dataclass
class OrbitalParams:
    semi_major_axis: float
    eccentricity: float
    inclination_deg: float
    angular_speed: float
    phase: float = 0.0


# =========================================================
# БАЗОВЫЕ УЗЛЫ СИСТЕМЫ
# =========================================================
@dataclass
class Body:
    id: int
    name: str
    body_type: BodyType
    mass: float
    radius: float


@dataclass
class SystemNode:
    name: str
    children: List["OrbitNode"] = field(default_factory=list)

    def get_world_position(self) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def add_child(self, child: "OrbitNode") -> None:
        self.children.append(child)
        child.parent = self


@dataclass
class OrbitNode:
    body: Body
    orbit: Optional[OrbitalParams] = None
    parent: Optional[Union["OrbitNode", "SystemNode", "BinarySystemRoot"]] = None
    children: List["OrbitNode"] = field(default_factory=list)

    def add_child(self, child: "OrbitNode") -> None:
        self.children.append(child)
        child.parent = self

    def get_world_position(self) -> np.ndarray:
        return np.zeros(3, dtype=float)


@dataclass
class BinarySystemRoot:
    name: str
    primary: OrbitNode
    secondary: OrbitNode
    children: List[OrbitNode] = field(default_factory=list)
    show_barycenter: bool = False  # для чистой звёздной пары не выпячиваем пустую точку

    def get_world_position(self) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def add_child(self, child: OrbitNode) -> None:
        self.children.append(child)
        child.parent = self


# =========================================================
# ID GENERATOR
# =========================================================
class IdGen:
    def __init__(self):
        self.value = 1

    def next(self) -> int:
        v = self.value
        self.value += 1
        return v


# =========================================================
# ГЕНЕРАТОР СИСТЕМЫ
# =========================================================
class SystemGenerator:
    def __init__(self, seed: int, presets: dict[BodyType, BodyPreset]):
        self.seed = seed
        self.rng = random.Random(seed)
        self.presets = presets
        self.ids = IdGen()

    # -------------------------
    # БАЗОВЫЕ МЕТОДЫ
    # -------------------------
    def create_body(self, body_type: BodyType, name: str) -> Body:
        preset = self.presets[body_type]
        return Body(
            id=self.ids.next(),
            name=name,
            body_type=body_type,
            mass=preset.mass.sample(self.rng),
            radius=preset.radius.sample(self.rng),
        )

    def create_orbit(
        self,
        body_type: BodyType,
        semi_major_axis: float,
        phase: Optional[float] = None,
        eccentricity: Optional[float] = None,
        inclination_deg: Optional[float] = None,
        angular_speed: Optional[float] = None,
    ) -> OrbitalParams:
        preset = self.presets[body_type]
        return OrbitalParams(
            semi_major_axis=semi_major_axis,
            eccentricity=(
                eccentricity if eccentricity is not None
                else preset.eccentricity.sample(self.rng)
            ),
            inclination_deg=(
                inclination_deg if inclination_deg is not None
                else preset.inclination_deg.sample(self.rng)
            ),
            angular_speed=(
                angular_speed if angular_speed is not None
                else preset.orbit_speed.sample(self.rng)
            ),
            phase=(
                phase if phase is not None
                else self.rng.uniform(0.0, 2.0 * math.pi)
            ),
        )

    @staticmethod
    def periapsis(a: float, e: float) -> float:
        return a * (1.0 - e)

    @staticmethod
    def apoapsis(a: float, e: float) -> float:
        return a * (1.0 + e)

    @staticmethod
    def orbit_clearance(radius_a: float, radius_b: float, factor: float = 4.0) -> float:
        return factor * (radius_a + radius_b)

    @staticmethod
    def hill_radius(planet_a: float, planet_e: float, planet_mass: float, parent_mass: float) -> float:
        return planet_a * (1.0 - planet_e) * ((planet_mass / (3.0 * parent_mass)) ** (1.0 / 3.0))

    @staticmethod
    def binary_distances_to_barycenter(m1: float, m2: float, separation: float) -> tuple[float, float]:
        total = m1 + m2
        r1 = separation * (m2 / total)
        r2 = separation * (m1 / total)
        return r1, r2

    # -------------------------
    # ЛУНЫ: распределение и capacity
    # -------------------------
    def estimate_moon_capacity(self, planet_node: OrbitNode, central_mass: float) -> int:
        if planet_node.orbit is None:
            return 0

        hill = self.hill_radius(
            planet_a=planet_node.orbit.semi_major_axis,
            planet_e=planet_node.orbit.eccentricity,
            planet_mass=planet_node.body.mass,
            parent_mass=central_mass,
        )

        moon_min_dist = max(
            self.presets["Moon"].orbit_distance.min,
            planet_node.body.radius * 3.0
        )
        moon_max_dist = min(
            self.presets["Moon"].orbit_distance.max,
            hill * 0.4
        )

        if moon_min_dist >= moon_max_dist:
            return 0

        usable_span = moon_max_dist - moon_min_dist
        avg_moon_radius = (self.presets["Moon"].radius.min + self.presets["Moon"].radius.max) * 0.5
        avg_gap = max(1.0, avg_moon_radius * 6.0)

        capacity = int(usable_span // avg_gap)
        return max(0, min(capacity, 8))

    def distribute_global_moons(
        self,
        planets: list[OrbitNode],
        total_moons: int,
        central_mass: float,
    ) -> dict[int, int]:
        if total_moons <= 0 or not planets:
            return {}

        capacities: dict[int, int] = {}
        eligible_planets: list[OrbitNode] = []

        for planet in planets:
            cap = self.estimate_moon_capacity(planet, central_mass)
            if cap > 0:
                capacities[planet.body.id] = cap
                eligible_planets.append(planet)

        if not eligible_planets:
            return {}

        allocation = {planet.body.id: 0 for planet in eligible_planets}

        remaining = total_moons
        idx = 0
        n = len(eligible_planets)

        while remaining > 0:
            planet = eligible_planets[idx % n]
            pid = planet.body.id

            if allocation[pid] < capacities[pid]:
                allocation[pid] += 1
                remaining -= 1

            if all(allocation[p.body.id] >= capacities[p.body.id] for p in eligible_planets):
                break

            idx += 1

        return allocation

    # -------------------------
    # ПОДБОР НЕПЕРЕСЕКАЮЩИХСЯ ОРБИТ
    # -------------------------
    def sample_non_intersecting_orbits(
        self,
        parent_mass: float,
        children_specs: list[tuple[BodyType, Body]],
        base_min_distance: float,
        max_distance: float,
        clearance_factor: float = 2.2,
        allow_expand_beyond_preset: bool = True,
        expansion_factor: float = 1.8,
        compactness: float = 0.18,
    ) -> list[OrbitalParams]:
        result: list[OrbitalParams] = []
        prev_apo = base_min_distance

        for idx, (body_type, body) in enumerate(children_specs):
            preset = self.presets[body_type]
            success = False

            prev_body_radius = children_specs[idx - 1][1].radius if idx > 0 else body.radius

            required_min_a = max(
                preset.orbit_distance.min,
                prev_apo + self.orbit_clearance(prev_body_radius, body.radius, clearance_factor)
            )

            if allow_expand_beyond_preset:
                effective_max_distance = max(
                    max_distance,
                    preset.orbit_distance.max,
                    required_min_a * expansion_factor
                )
            else:
                effective_max_distance = min(max_distance, preset.orbit_distance.max)

            if required_min_a >= effective_max_distance:
                raise RuntimeError(
                    f"Не хватает диапазона орбиты для {body.name}: "
                    f"required_min_a={required_min_a:.2f}, "
                    f"effective_max_distance={effective_max_distance:.2f}"
                )

            local_span = max(5.0, (effective_max_distance - required_min_a) * compactness)
            local_max_a = min(effective_max_distance, required_min_a + local_span)

            for _ in range(500):
                e = preset.eccentricity.sample(self.rng)
                inc = preset.inclination_deg.sample(self.rng)
                speed = preset.orbit_speed.sample(self.rng)
                phase = self.rng.uniform(0.0, 2.0 * math.pi)

                a = self.rng.uniform(required_min_a, local_max_a)

                q = self.periapsis(a, e)
                Q = self.apoapsis(a, e)

                if q <= prev_apo:
                    continue

                orbit = OrbitalParams(
                    semi_major_axis=a,
                    eccentricity=e,
                    inclination_deg=inc,
                    angular_speed=speed,
                    phase=phase,
                )
                result.append(orbit)
                prev_apo = Q
                success = True
                break

            if not success:
                raise RuntimeError(f"Не удалось подобрать непересекающуюся орбиту для {body.name}")

        return result

    # -------------------------
    # ОДИНАРНЫЙ КОРЕНЬ
    # -------------------------
    def generate_single_root_system(
        self,
        counts: dict[BodyType, int],
        root_type: BodyType,
    ) -> SystemNode:
        root_body = self.create_body(root_type, f"{root_type}_Root")
        root = SystemNode(name=f"{root_body.name}_System")

        root_body_node = OrbitNode(body=root_body, orbit=None, parent=root)
        root.add_child(root_body_node)

        star_count = counts.get("Star", 0)
        planet_count = counts.get("Planet", 0)
        moon_count_total = counts.get("Moon", 0)

        if root_type == "Star":
            star_count = max(0, star_count - 1)

        star_nodes: list[OrbitNode] = []

        if star_count > 0:
            star_bodies = [self.create_body("Star", f"Star_{i + 1}") for i in range(star_count)]

            star_specs = sorted(
                [("Star", body) for body in star_bodies],
                key=lambda x: x[1].mass,
                reverse=True
            )

            star_orbits = self.sample_non_intersecting_orbits(
                parent_mass=root_body.mass,
                children_specs=star_specs,
                base_min_distance=root_body.radius * 8.0,
                max_distance=max(
                    self.presets["Star"].orbit_distance.max,
                    root_body.radius * 18.0
                ),
                clearance_factor=2.0,
                allow_expand_beyond_preset=True,
                expansion_factor=1.6,
                compactness=0.12,
            )

            for (_, body), orbit in zip(star_specs, star_orbits):
                node = OrbitNode(body=body, orbit=orbit, parent=root)
                root.add_child(node)
                star_nodes.append(node)

        planet_nodes: list[OrbitNode] = []

        if planet_count > 0:
            planet_bodies = [self.create_body("Planet", f"Planet_{i + 1}") for i in range(planet_count)]

            planet_specs = sorted(
                [("Planet", body) for body in planet_bodies],
                key=lambda x: x[1].mass,
                reverse=True
            )

            base_min_distance = root_body.radius * 14.0

            if star_nodes:
                stars_outer_edge = max(
                    self.apoapsis(node.orbit.semi_major_axis, node.orbit.eccentricity)
                    for node in star_nodes
                    if node.orbit is not None
                )
                base_min_distance = max(
                    base_min_distance,
                    stars_outer_edge + root_body.radius * 8.0
                )

            planet_orbits = self.sample_non_intersecting_orbits(
                parent_mass=root_body.mass,
                children_specs=planet_specs,
                base_min_distance=base_min_distance,
                max_distance=max(
                    self.presets["Planet"].orbit_distance.max,
                    base_min_distance * 2.2
                ),
                clearance_factor=2.0,
                allow_expand_beyond_preset=True,
                expansion_factor=1.6,
                compactness=0.10,
            )

            for (_, body), orbit in zip(planet_specs, planet_orbits):
                node = OrbitNode(body=body, orbit=orbit, parent=root)
                root.add_child(node)
                planet_nodes.append(node)

        self.attach_moons_to_planets(
            planets=planet_nodes,
            central_mass=root_body.mass,
            total_moons=moon_count_total,
        )

        return root

    # -------------------------
    # БИНАРНЫЙ КОРЕНЬ
    # -------------------------
    def generate_binary_root_system(
        self,
        counts: dict[BodyType, int],
        primary_type: BodyType,
        secondary_type: BodyType,
    ) -> BinarySystemRoot:
        primary_body = self.create_body(primary_type, f"{primary_type}_Primary")
        secondary_body = self.create_body(secondary_type, f"{secondary_type}_Secondary")

        if primary_type == "Star":
            counts["Star"] = max(0, counts.get("Star", 0) - 1)
        if secondary_type == "Star":
            counts["Star"] = max(0, counts.get("Star", 0) - 1)
        if primary_type == "BH":
            counts["BH"] = max(0, counts.get("BH", 0) - 1)
        if secondary_type == "BH":
            counts["BH"] = max(0, counts.get("BH", 0) - 1)

        planet_count = counts.get("Planet", 0)
        moon_count_total = counts.get("Moon", 0)

        # Для двух звезд по умолчанию делаем КРУГОВУЮ бинарную пару,
        # чтобы не было уродских отдельных эллипсов вокруг красного крестика.
        # Физически это всё равно орбита вокруг общего центра масс, просто e=0.
        is_star_star_binary = primary_type == "Star" and secondary_type == "Star"

        min_sep = (primary_body.radius + secondary_body.radius) * 8.0
        preset_max_sep = max(
            self.presets[primary_type].orbit_distance.max if primary_type in self.presets else min_sep * 2.0,
            self.presets[secondary_type].orbit_distance.max if secondary_type in self.presets else min_sep * 2.0,
            min_sep * 2.0
        )
        max_sep = min(preset_max_sep, min_sep * 4.0)
        separation = self.rng.uniform(min_sep, max_sep)

        r1, r2 = self.binary_distances_to_barycenter(
            primary_body.mass,
            secondary_body.mass,
            separation
        )

        if is_star_star_binary:
            pair_e = 0.0
            pair_inc = self.rng.uniform(0.0, 6.0)
            pair_speed = self.rng.uniform(0.015, 0.04)
        else:
            pair_e = self.rng.uniform(0.0, 0.20)
            pair_inc = self.rng.uniform(0.0, 20.0)
            pair_speed = self.rng.uniform(0.01, 0.05)

        pair_phase = self.rng.uniform(0.0, 2.0 * math.pi)

        primary_node = OrbitNode(
            body=primary_body,
            orbit=self.create_orbit(
                primary_type,
                semi_major_axis=r1,
                eccentricity=pair_e,
                inclination_deg=pair_inc,
                angular_speed=pair_speed,
                phase=pair_phase,
            ),
        )

        secondary_node = OrbitNode(
            body=secondary_body,
            orbit=self.create_orbit(
                secondary_type,
                semi_major_axis=r2,
                eccentricity=pair_e,
                inclination_deg=pair_inc,
                angular_speed=pair_speed,
                phase=pair_phase + math.pi,
            ),
        )

        root = BinarySystemRoot(
            name="BinaryBarycenter",
            primary=primary_node,
            secondary=secondary_node,
            show_barycenter=not is_star_star_binary,  # для star-star не рисуем центр как “пустую штуку”
        )

        primary_node.parent = root
        secondary_node.parent = root

        binary_outer_min = max(
            self.apoapsis(primary_node.orbit.semi_major_axis, primary_node.orbit.eccentricity),
            self.apoapsis(secondary_node.orbit.semi_major_axis, secondary_node.orbit.eccentricity),
        ) + self.orbit_clearance(primary_body.radius, secondary_body.radius, 8.0)

        extra_star_count = counts.get("Star", 0)
        extra_star_nodes: list[OrbitNode] = []

        if extra_star_count > 0:
            extra_bodies = [self.create_body("Star", f"Star_Extra_{i + 1}") for i in range(extra_star_count)]

            extra_specs = sorted(
                [("Star", body) for body in extra_bodies],
                key=lambda x: x[1].mass,
                reverse=True
            )

            extra_orbits = self.sample_non_intersecting_orbits(
                parent_mass=primary_body.mass + secondary_body.mass,
                children_specs=extra_specs,
                base_min_distance=binary_outer_min,
                max_distance=max(
                    self.presets["Star"].orbit_distance.max * 1.3,
                    binary_outer_min * 1.8
                ),
                clearance_factor=2.0,
                allow_expand_beyond_preset=True,
                expansion_factor=1.5,
                compactness=0.10,
            )

            for (_, body), orbit in zip(extra_specs, extra_orbits):
                node = OrbitNode(body=body, orbit=orbit, parent=root)
                root.add_child(node)
                extra_star_nodes.append(node)

        planets_base_min = binary_outer_min

        if extra_star_nodes:
            extra_stars_outer_edge = max(
                self.apoapsis(node.orbit.semi_major_axis, node.orbit.eccentricity)
                for node in extra_star_nodes
                if node.orbit is not None
            )
            planets_base_min = max(
                planets_base_min,
                extra_stars_outer_edge + max(primary_body.radius, secondary_body.radius) * 8.0
            )

        planet_nodes: list[OrbitNode] = []

        if planet_count > 0:
            planet_bodies = [self.create_body("Planet", f"Planet_{i + 1}") for i in range(planet_count)]

            planet_specs = sorted(
                [("Planet", body) for body in planet_bodies],
                key=lambda x: x[1].mass,
                reverse=True
            )

            planet_orbits = self.sample_non_intersecting_orbits(
                parent_mass=primary_body.mass + secondary_body.mass,
                children_specs=planet_specs,
                base_min_distance=planets_base_min,
                max_distance=max(
                    self.presets["Planet"].orbit_distance.max * 1.5,
                    planets_base_min * 2.0
                ),
                clearance_factor=2.0,
                allow_expand_beyond_preset=True,
                expansion_factor=1.6,
                compactness=0.08,
            )

            for (_, body), orbit in zip(planet_specs, planet_orbits):
                node = OrbitNode(body=body, orbit=orbit, parent=root)
                root.add_child(node)
                planet_nodes.append(node)

        self.attach_moons_to_planets(
            planets=planet_nodes,
            central_mass=primary_body.mass + secondary_body.mass,
            total_moons=moon_count_total,
        )

        return root

    # -------------------------
    # ЛУНЫ
    # -------------------------
    def attach_moons_to_planets(
        self,
        planets: list[OrbitNode],
        central_mass: float,
        total_moons: int,
    ) -> None:
        if total_moons <= 0 or not planets:
            return

        moon_distribution = self.distribute_global_moons(
            planets=planets,
            total_moons=total_moons,
            central_mass=central_mass,
        )

        if not moon_distribution:
            return

        moon_index = 1

        for planet_node in planets:
            assigned_moons = moon_distribution.get(planet_node.body.id, 0)
            if assigned_moons <= 0 or planet_node.orbit is None:
                continue

            hill = self.hill_radius(
                planet_a=planet_node.orbit.semi_major_axis,
                planet_e=planet_node.orbit.eccentricity,
                planet_mass=planet_node.body.mass,
                parent_mass=central_mass,
            )

            moon_bodies = [
                self.create_body("Moon", f"Moon_{moon_index + i}")
                for i in range(assigned_moons)
            ]
            moon_index += assigned_moons

            moon_specs = sorted(
                [("Moon", body) for body in moon_bodies],
                key=lambda x: x[1].mass,
                reverse=True
            )

            moon_orbits = self.sample_non_intersecting_moon_orbits(
                moon_specs=moon_specs,
                planet_radius=planet_node.body.radius,
                hill_radius_limit=hill,
            )

            for (_, body), orbit in zip(moon_specs, moon_orbits):
                moon_node = OrbitNode(body=body, orbit=orbit, parent=planet_node)
                planet_node.add_child(moon_node)

    def sample_non_intersecting_moon_orbits(
        self,
        moon_specs: list[tuple[BodyType, Body]],
        planet_radius: float,
        hill_radius_limit: float,
    ) -> list[OrbitalParams]:
        result: list[OrbitalParams] = []
        placed_ranges: list[tuple[float, float, float]] = []

        current_min = max(
            self.presets["Moon"].orbit_distance.min,
            planet_radius * 3.0
        )

        max_allowed = min(
            self.presets["Moon"].orbit_distance.max,
            hill_radius_limit * 0.4
        )

        if current_min >= max_allowed:
            return result

        for _, moon_body in moon_specs:
            success = False

            inner_gap = max(
                0.6,
                moon_body.radius * 4.0
            )

            required_min_a = current_min + inner_gap

            if required_min_a >= max_allowed:
                break

            for _ in range(400):
                e = self.rng.uniform(0.0, min(0.03, self.presets["Moon"].eccentricity.max))
                inc = self.presets["Moon"].inclination_deg.sample(self.rng)
                speed = self.presets["Moon"].orbit_speed.sample(self.rng)
                phase = self.rng.uniform(0.0, 2.0 * math.pi)

                local_span = max(0.8, (max_allowed - required_min_a) * 0.08)
                local_max_a = min(max_allowed, required_min_a + local_span)

                if required_min_a >= local_max_a:
                    continue

                a = self.rng.uniform(required_min_a, local_max_a)

                q = self.periapsis(a, e)
                Q = self.apoapsis(a, e)

                intersects = False
                for existing_q, existing_Q, existing_radius in placed_ranges:
                    safety_gap = max(
                        0.8,
                        (existing_radius + moon_body.radius) * 3.5
                    )
                    if not (Q + safety_gap < existing_q or q > existing_Q + safety_gap):
                        intersects = True
                        break

                if intersects:
                    continue

                orbit = OrbitalParams(
                    semi_major_axis=a,
                    eccentricity=e,
                    inclination_deg=inc,
                    angular_speed=speed,
                    phase=phase,
                )

                result.append(orbit)
                placed_ranges.append((q, Q, moon_body.radius))
                current_min = Q + max(1.0, moon_body.radius * 4.0)

                success = True
                break

            if not success:
                break

        return result

    # -------------------------
    # ОСНОВНОЙ generate_system()
    # -------------------------
    def generate_system(self, counts: dict[BodyType, int]):
        bh_count = counts.get("BH", 0)
        star_count = counts.get("Star", 0)

        if bh_count <= 0 and star_count <= 0:
            raise ValueError("Нужен хотя бы один центральный объект: BH или Star")

        counts = dict(counts)

        if counts.get("BH", 0) >= 2:
            return self.generate_binary_root_system(counts, "BH", "BH")

        if counts.get("BH", 0) == 1 and counts.get("Star", 0) >= 1:
            return self.generate_binary_root_system(counts, "BH", "Star")

        if counts.get("Star", 0) >= 2:
            return self.generate_binary_root_system(counts, "Star", "Star")

        if counts.get("BH", 0) == 1:
            return self.generate_single_root_system(counts, "BH")

        return self.generate_single_root_system(counts, "Star")


# =========================================================
# FLATTEN NODES
# =========================================================
def flatten_nodes(root) -> list[OrbitNode]:
    result: list[OrbitNode] = []

    def dfs(node):
        if isinstance(node, OrbitNode):
            result.append(node)
            for child in node.children:
                dfs(child)

        elif isinstance(node, SystemNode):
            for child in node.children:
                dfs(child)

        elif isinstance(node, BinarySystemRoot):
            dfs(node.primary)
            dfs(node.secondary)
            for child in node.children:
                dfs(child)

    dfs(root)
    return result


# =========================================================
# МАТЕМАТИКА ДЛЯ ВИЗУАЛИЗАЦИИ
# =========================================================
def solve_kepler(M: float, e: float, tol: float = 1e-10, max_iter: int = 30) -> float:
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
    E = solve_kepler(M, e)

    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E)

    inc = np.radians(inc_deg)

    x = x_orb
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)

    return np.array([x, y, z], dtype=float)


def kepler_local_curve(a: float, e: float, inc_deg: float, num: int = 400):
    E_vals = np.linspace(0, 2 * np.pi, num)

    x_orb = a * (np.cos(E_vals) - e)
    y_orb = a * np.sqrt(1 - e ** 2) * np.sin(E_vals)

    inc = np.radians(inc_deg)

    x = x_orb
    y = y_orb * np.cos(inc)
    z = y_orb * np.sin(inc)

    return x, y, z


# =========================================================
# ВИЗУАЛИЗАТОР
# =========================================================
class VisualOrbitalBody:
    def __init__(self, node: OrbitNode):
        self.node = node
        self.point_artist = None
        self.orbit_artist = None
        self.current_mean_anomaly = node.orbit.phase if node.orbit is not None else 0.0

    def body_type(self) -> BodyType:
        return self.node.body.body_type

    def color(self) -> str:
        mapping = {
            "BH": "black",
            "Star": "gold",
            "Planet": "deepskyblue",
            "Moon": "lightgray",
        }
        return mapping[self.node.body.body_type]

    def marker_size(self) -> int:
        mapping = {
            "BH": 12,
            "Star": 9,
            "Planet": 5,
            "Moon": 3,
        }
        return mapping[self.node.body.body_type]

    def create_artists(self, ax):
        self.point_artist, = ax.plot(
            [], [], [],
            color=self.color(),
            marker="o",
            linestyle="None",
            markersize=self.marker_size()
        )

        if self.node.orbit is not None:
            self.orbit_artist, = ax.plot(
                [], [], [],
                color=self.color(),
                linestyle="--",
                linewidth=1,
                alpha=0.75
            )

    def update(self, dt: float, speed_scale: dict[BodyType, float]):
        if self.node.orbit is not None:
            self.current_mean_anomaly += self.node.orbit.angular_speed * speed_scale[self.body_type()] * dt

    def get_parent_world_position(
        self,
        root,
        visual_map: dict[int, "VisualOrbitalBody"],
        distance_scale: dict[BodyType, float],
    ) -> np.ndarray:
        parent = self.node.parent

        if parent is None:
            return np.zeros(3, dtype=float)

        if isinstance(parent, (SystemNode, BinarySystemRoot)):
            return np.zeros(3, dtype=float)

        if isinstance(parent, OrbitNode):
            parent_visual = visual_map[parent.body.id]
            return parent_visual.get_world_position(root, visual_map, distance_scale)

        return np.zeros(3, dtype=float)

    def get_world_position(
        self,
        root,
        visual_map: dict[int, "VisualOrbitalBody"],
        distance_scale: dict[BodyType, float],
    ) -> np.ndarray:
        parent_pos = self.get_parent_world_position(root, visual_map, distance_scale)

        if self.node.orbit is None:
            return parent_pos

        scaled_a = self.node.orbit.semi_major_axis * distance_scale[self.body_type()]

        local = kepler_local_position(
            a=scaled_a,
            e=self.node.orbit.eccentricity,
            inc_deg=self.node.orbit.inclination_deg,
            M=self.current_mean_anomaly,
        )
        return parent_pos + local

    def get_world_orbit_curve(
        self,
        root,
        visual_map: dict[int, "VisualOrbitalBody"],
        distance_scale: dict[BodyType, float],
    ):
        parent_pos = self.get_parent_world_position(root, visual_map, distance_scale)

        if self.node.orbit is None:
            return None

        scaled_a = self.node.orbit.semi_major_axis * distance_scale[self.body_type()]

        x, y, z = kepler_local_curve(
            a=scaled_a,
            e=self.node.orbit.eccentricity,
            inc_deg=self.node.orbit.inclination_deg
        )

        return x + parent_pos[0], y + parent_pos[1], z + parent_pos[2]

    def refresh_artists(
        self,
        root,
        visual_map: dict[int, "VisualOrbitalBody"],
        distance_scale: dict[BodyType, float],
    ):
        pos = self.get_world_position(root, visual_map, distance_scale)

        self.point_artist.set_data([pos[0]], [pos[1]])
        self.point_artist.set_3d_properties([pos[2]])

        if self.orbit_artist is not None:
            curve = self.get_world_orbit_curve(root, visual_map, distance_scale)
            if curve is not None:
                x, y, z = curve
                self.orbit_artist.set_data(x, y)
                self.orbit_artist.set_3d_properties(z)


# =========================================================
# ПЕЧАТЬ СТРУКТУРЫ
# =========================================================
def print_system(root, indent=0):
    prefix = "  " * indent

    if isinstance(root, SystemNode):
        print(f"{prefix}[System] {root.name}")
        for child in root.children:
            print_system(child, indent + 1)

    elif isinstance(root, BinarySystemRoot):
        print(f"{prefix}[BinaryRoot] {root.name} show_barycenter={root.show_barycenter}")
        print_system(root.primary, indent + 1)
        print_system(root.secondary, indent + 1)
        for child in root.children:
            print_system(child, indent + 1)

    elif isinstance(root, OrbitNode):
        orbit_str = ""
        if root.orbit is not None:
            orbit_str = (
                f" | a={root.orbit.semi_major_axis:.2f}"
                f" e={root.orbit.eccentricity:.2f}"
                f" inc={root.orbit.inclination_deg:.2f}"
                f" speed={root.orbit.angular_speed:.4f}"
            )
        print(
            f"{prefix}[{root.body.body_type}] {root.body.name}"
            f" mass={root.body.mass:.2f}"
            f" radius={root.body.radius:.2f}"
            f"{orbit_str}"
        )
        for child in root.children:
            print_system(child, indent + 1)


# =========================================================
# ПРЕСЕТЫ
# =========================================================
presets = {
    "BH": BodyPreset(
        body_type="BH",
        mass=RangeF(5000.0, 20000.0),
        radius=RangeF(3.0, 8.0),
        eccentricity=RangeF(0.0, 0.08),
        inclination_deg=RangeF(0.0, 10.0),
        orbit_distance=RangeF(20.0, 100.0),
        orbit_speed=RangeF(0.01, 0.03),
        moon_count=RangeI(0, 0),
    ),
    "Star": BodyPreset(
        body_type="Star",
        mass=RangeF(100.0, 500.0),
        radius=RangeF(1.2, 4.0),
        eccentricity=RangeF(0.0, 0.25),
        inclination_deg=RangeF(0.0, 20.0),
        orbit_distance=RangeF(20.0, 120.0),
        orbit_speed=RangeF(0.01, 0.05),
        moon_count=RangeI(0, 0),
    ),
    "Planet": BodyPreset(
        body_type="Planet",
        mass=RangeF(1.0, 20.0),
        radius=RangeF(0.4, 2.2),
        eccentricity=RangeF(0.0, 0.15),
        inclination_deg=RangeF(0.0, 18.0),
        orbit_distance=RangeF(40.0, 260.0),
        orbit_speed=RangeF(0.003, 0.02),
        moon_count=RangeI(0, 4),
    ),
    "Moon": BodyPreset(
        body_type="Moon",
        mass=RangeF(0.01, 0.2),
        radius=RangeF(0.08, 0.4),
        eccentricity=RangeF(0.0, 0.08),
        inclination_deg=RangeF(0.0, 20.0),
        orbit_distance=RangeF(1.5, 16.0),
        orbit_speed=RangeF(0.04, 0.18),
        moon_count=RangeI(0, 0),
    ),
}


# =========================================================
# НАСТРОЙКИ ГЕНЕРАЦИИ
# =========================================================
seed = 1337

counts = {
    "BH": 2,
    "Star": 3,
    "Planet": 5,
    "Moon": 6,
}

generator = SystemGenerator(seed=seed, presets=presets)
root = generator.generate_system(counts)

print_system(root)

nodes = flatten_nodes(root)
visual_bodies = [VisualOrbitalBody(node) for node in nodes]
visual_map = {vb.node.body.id: vb for vb in visual_bodies}


# =========================================================
# FIGURE
# =========================================================
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.30)

ax.set_title("Generated Star System")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
ax.set_zlim(-300, 300)

try:
    ax.set_box_aspect([1, 1, 1])
except Exception:
    pass

ax.view_init(elev=24, azim=35)

# Барицентр рисуем только там, где он реально нужен как визуальный ориентир
show_bary = not (isinstance(root, BinarySystemRoot) and not root.show_barycenter)
if show_bary:
    bary_artist, = ax.plot([0], [0], [0], "r+", markersize=12)
else:
    bary_artist, = ax.plot([], [], [], "r+", markersize=0)

for vb in visual_bodies:
    vb.create_artists(ax)


# =========================================================
# СЛАЙДЕРЫ
# =========================================================
slider_left = 0.18
slider_width = 0.67
slider_height = 0.025
slider_gap = 0.04
start_y = 0.20

axes_sliders = []
for i in range(6):
    y = start_y - i * slider_gap
    axes_sliders.append(fig.add_axes((slider_left, y, slider_width, slider_height)))

slider_global_speed = Slider(axes_sliders[0], "global speed", 0.1, 5.0, valinit=1.0)
slider_star_dist = Slider(axes_sliders[1], "star distance", 0.5, 2.5, valinit=1.0)
slider_planet_dist = Slider(axes_sliders[2], "planet distance", 0.5, 2.5, valinit=1.0)
slider_moon_dist = Slider(axes_sliders[3], "moon distance", 0.5, 3.0, valinit=1.0)
slider_planet_speed = Slider(axes_sliders[4], "planet speed", 0.2, 3.0, valinit=1.0)
slider_moon_speed = Slider(axes_sliders[5], "moon speed", 0.2, 5.0, valinit=1.0)


# =========================================================
# INIT / UPDATE
# =========================================================
def get_distance_scale() -> dict[BodyType, float]:
    return {
        "BH": 1.0,
        "Star": slider_star_dist.val,
        "Planet": slider_planet_dist.val,
        "Moon": slider_moon_dist.val,
    }


def get_speed_scale() -> dict[BodyType, float]:
    return {
        "BH": 1.0,
        "Star": 1.0,
        "Planet": slider_planet_speed.val,
        "Moon": slider_moon_speed.val,
    }


def init():
    distance_scale = get_distance_scale()

    artists = [bary_artist]
    for vb in visual_bodies:
        vb.refresh_artists(root, visual_map, distance_scale)
        artists.append(vb.point_artist)
        if vb.orbit_artist is not None:
            artists.append(vb.orbit_artist)

    return artists


def update(frame):
    global_speed = slider_global_speed.val
    distance_scale = get_distance_scale()
    speed_scale = get_speed_scale()

    dt = global_speed

    artists = [bary_artist]

    for vb in visual_bodies:
        vb.update(dt, speed_scale)

    for vb in visual_bodies:
        vb.refresh_artists(root, visual_map, distance_scale)
        artists.append(vb.point_artist)
        if vb.orbit_artist is not None:
            artists.append(vb.orbit_artist)

    return artists


ani = FuncAnimation(
    fig,
    update,
    init_func=init,
    interval=40,
    blit=False,
    cache_frame_data=False,
)

plt.show()