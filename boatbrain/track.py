import math

import numpy as np

from .config import Config

TRACK_INFO = {
    "oval": ("Oval", "Gentle training loop"),
    "harbor": ("Harbor", "Uneven coastal bends"),
    "technical": ("Technical", "Mixed-radius test circuit"),
    "speedway": ("Speedway", "Long straights and rounded corners"),
    "serpentine": ("Serpentine", "Flowing S bends and changing curvature"),
    "switchback": ("Switchback", "Mountain descent / point to point"),
    "slalom": ("Slalom", "Wide circuit with alternating barricades"),
    "babft": ("BABFT Run", "6,000-unit obstacle river / point to point"),
}
TRACKS = tuple(TRACK_INFO)


class Track:
    def __init__(self, name="oval", config=Config()):
        if name not in TRACKS:
            raise ValueError(f"Unknown track {name}")
        self.name, self.config = name, config
        self.title, self.description = TRACK_INFO[name]
        self.closed = name not in ("switchback", "babft")
        self.width, self.height = config.width, config.height
        self.road_half_width = config.road_half_width
        self.obstacles = []  # (x, y, width, height) in world units.
        t = np.linspace(0, 2 * math.pi, 160, endpoint=False)
        x, y = 480 + 345 * np.cos(t), 360 + 225 * np.sin(t)
        if name == "harbor":
            x += 22 * np.sin(2 * t)
            y += 30 * np.cos(3 * t)
        elif name == "technical":
            x += 32 * np.sin(3 * t)
            y += 42 * np.sin(2 * t)
        elif name == "speedway":
            x = 480 + 335 * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** 0.65
            y = 360 + 215 * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** 0.65
            self.road_half_width = 62
        elif name == "serpentine":
            radius = 1 + 0.14 * np.cos(3 * t)
            x = 480 + 310 * radius * np.cos(t)
            y = 360 + 205 * radius * np.sin(t)
            self.road_half_width = 50
        elif name == "slalom":
            x, y = 480 + 320 * np.cos(t), 360 + 200 * np.sin(t)
            self.road_half_width = 84
            for i, a in enumerate((0.45, 0.95, 2.0, 2.55, 3.6, 4.15, 4.75, 5.35)):
                center = np.array([480 + 320 * math.cos(a), 360 + 200 * math.sin(a)])
                tangent = np.array([-320 * math.sin(a), 200 * math.cos(a)])
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                center += normal * (30 if i % 2 else -30)
                self.obstacles.append((center[0] - 16, center[1] - 16, 32, 32))
        elif name == "switchback":
            self.height, self.road_half_width = 2720, 66
            y = np.linspace(2520, 200, 180)
            x = 480 + 250 * np.sin((2520 - y) / 250)
        elif name == "babft":
            self.height, self.road_half_width = 6400, 190
            y = np.array([6200.0, 200.0])
            x = np.array([480.0, 480.0])
            for i, yy in enumerate(range(5900, 350, -280)):
                kind = i % 4
                if kind == 0:
                    self.obstacles.append((290, yy, 174, 62))
                elif kind == 1:
                    self.obstacles.append((496, yy, 174, 62))
                elif kind == 2:
                    self.obstacles.append((438, yy, 84, 76))
                else:
                    self.obstacles.extend([(290, yy, 102, 70), (568, yy, 102, 70)])
        self.points = np.column_stack((x, y))
        self.segment_starts = self.points if self.closed else self.points[:-1]
        self.delta = (
            np.roll(self.points, -1, axis=0) - self.points
            if self.closed
            else np.diff(self.points, axis=0)
        )
        self.lengths = np.linalg.norm(self.delta, axis=1)
        self.cumulative = np.r_[0, np.cumsum(self.lengths)]
        self.length = self.cumulative[-1]
        gx, gy = np.meshgrid(
            np.arange(self.width // config.cell), np.arange(self.height // config.cell)
        )
        centers = np.column_stack(
            ((gx.ravel() + 0.5) * config.cell, (gy.ravel() + 0.5) * config.cell)
        )
        # Limit temporary distance arrays on long maps.
        distances = [
            self.project(centers[i : i + 4000])[1] for i in range(0, len(centers), 4000)
        ]
        self.wall = (np.concatenate(distances) > self.road_half_width).reshape(gx.shape)
        for ox, oy, ow, oh in self.obstacles:
            # Round outward to include boundary cells.
            left, top = int(ox // config.cell), int(oy // config.cell)
            right, bottom = (
                math.ceil((ox + ow) / config.cell),
                math.ceil((oy + oh) / config.cell),
            )
            self.wall[max(0, top) : bottom, max(0, left) : right] = True
        self._sensor_cache = {}

    def scenarios(self):
        if self.closed:
            return [(0.0, 1, 0), (0.25, 1, 12), (0.5, -1, -12)]
        return [(0.0, 1, 0), (0.0, 1, 24), (1.0, -1, -24)]

    def project(self, positions):
        d = np.asarray(positions)[:, None, :] - self.segment_starts[None, :, :]
        u = np.clip(np.sum(d * self.delta, axis=2) / self.lengths**2, 0, 1)
        residual = d - u[:, :, None] * self.delta
        dsq = np.sum(residual**2, axis=2)
        nearest = np.argmin(dsq, axis=1)
        rows = np.arange(len(positions))
        return (
            self.cumulative[nearest] + u[rows, nearest] * self.lengths[nearest],
            np.sqrt(dsq[rows, nearest]),
            nearest,
        )

    def spawn(self, fraction=0.0, direction=1, offset=0):
        s = ((fraction % 1) if self.closed else np.clip(fraction, 0, 1)) * self.length
        i = min(
            np.searchsorted(self.cumulative, s, side="right") - 1, len(self.delta) - 1
        )
        v = self.delta[i] / self.lengths[i]
        position = (
            self.segment_starts[i]
            + v * (s - self.cumulative[i])
            + np.array([-v[1], v[0]]) * offset
        )
        return position, math.atan2(v[1], v[0]) + (math.pi if direction == -1 else 0)

    def occupied(self, xy):
        grid = np.floor(np.asarray(xy) / self.config.cell).astype(int)
        x, y = grid[..., 0], grid[..., 1]
        outside = (
            (x < 0) | (y < 0) | (x >= self.wall.shape[1]) | (y >= self.wall.shape[0])
        )
        return (
            outside
            | self.wall[
                np.clip(y, 0, self.wall.shape[0] - 1),
                np.clip(x, 0, self.wall.shape[1] - 1),
            ]
        )

    def sensor_offsets(self, heading):
        # Separating-axis test against cells at the quantized heading.
        if heading in self._sensor_cache:
            return self._sensor_cache[heading]
        c = self.config
        angle = heading * 2 * math.pi / c.heading_bins
        axis = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-axis[1], axis[0]])
        length, half = c.sensor_lengths[-1], c.sensor_width / 2
        radius = math.ceil((length + c.sensor_width) / c.cell)
        xx, yy = np.meshgrid(
            np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)
        )
        offsets = np.column_stack((xx.ravel(), yy.ravel()))
        centers = offsets * c.cell
        along, across = centers @ axis, centers @ normal
        pad = c.cell / 2 * (abs(axis[0]) + abs(axis[1]))
        difference = abs(centers - axis * length / 2)
        extents = abs(axis) * length / 2 + abs(normal) * half + c.cell / 2
        hit = (
            (along >= -pad)
            & (along <= length + pad)
            & (abs(across) <= half + pad)
            & np.all(difference <= extents, axis=1)
        )
        offsets, centers, along = offsets[hit], centers[hit], along[hit]
        masks = []
        for distance in c.sensor_lengths:
            difference = abs(centers - axis * distance / 2)
            extents = abs(axis) * distance / 2 + abs(normal) * half + c.cell / 2
            masks.append(
                (along <= distance + pad) & np.all(difference <= extents, axis=1)
            )
        result = offsets, np.stack(masks, axis=1)
        self._sensor_cache[heading] = result
        return result

    def sense(self, positions, angles):
        c = self.config
        origins = np.floor(np.asarray(positions) / c.cell).astype(int)
        output = np.zeros((len(origins), c.inputs), dtype=np.int8)
        for beam, relative in enumerate(c.sensor_angles):
            headings = (
                np.rint(
                    (angles + math.radians(relative)) * c.heading_bins / (2 * math.pi)
                ).astype(int)
                % c.heading_bins
            )
            for heading in np.unique(headings):
                rows = np.flatnonzero(headings == heading)
                offsets, masks = self.sensor_offsets(int(heading))
                grid = origins[rows, None, :] + offsets[None, :, :]
                occupied = self.occupied((grid + 0.5) * c.cell)
                values = np.any(occupied[:, :, None] & masks[None, :, :], axis=1)
                j = beam * len(c.sensor_lengths)
                output[rows, j : j + len(c.sensor_lengths)] = values
        return output

    def sensor_polygon(self, position, angle, relative, length):
        c = self.config
        origin = (np.floor(np.asarray(position) / c.cell) + 0.5) * c.cell
        h = (
            round((angle + math.radians(relative)) * c.heading_bins / (2 * math.pi))
            % c.heading_bins
        )
        a = h * 2 * math.pi / c.heading_bins
        axis = np.array([math.cos(a), math.sin(a)])
        normal = np.array([-axis[1], axis[0]]) * c.sensor_width / 2
        return [
            origin - normal,
            origin + axis * length - normal,
            origin + axis * length + normal,
            origin + normal,
        ]
