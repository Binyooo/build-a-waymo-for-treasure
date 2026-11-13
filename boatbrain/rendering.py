import math

import numpy as np
import pygame
import pygame.gfxdraw

GROUND = (17, 28, 36)
ROAD = (43, 55, 68)


def polygon(surface, color, points):
    points = [(round(p[0]), round(p[1])) for p in points]
    if len(points) >= 3:
        pygame.draw.polygon(surface, color, points)
        pygame.gfxdraw.aapolygon(surface, points, color)


class WorldView:
    def __init__(self, track, size):
        self.track = track
        self.size = tuple(size)
        self.surface = pygame.Surface(size)
        self.camera = np.zeros(2)
        self.scale = 1.0
        self.zoom = 1.0
        self._cache_key = None
        self._static = None
        self.font = pygame.font.SysFont("Menlo,Consolas,monospace", 14)
        unit = track.delta / track.lengths[:, None]
        if track.closed:
            normals = unit + np.roll(unit, 1, axis=0)
        else:
            normals = np.vstack((unit[0], unit[:-1] + unit[1:], unit[-1]))
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        self.normals = np.column_stack((-normals[:, 1], normals[:, 0]))

    def frame(self, position, angle):
        t = self.track
        if t.closed:
            self.scale = (
                min(self.size[0] / t.width, self.size[1] / t.height) * 0.96 * self.zoom
            )
            self.camera = np.array([t.width / 2, t.height / 2])
        else:
            self.scale = 1.7 * self.zoom
            lookahead = np.array([math.cos(angle), math.sin(angle)]) * 100
            self.camera = np.asarray(position) + lookahead
            half = np.array(self.size) / (2 * self.scale)
            self.camera = np.maximum(
                half, np.minimum([t.width, t.height] - half, self.camera)
            )
        key = (round(self.camera[0], 1), round(self.camera[1], 1), self.scale)
        if key != self._cache_key:
            self._cache_key = key
            self._static = self.background()
        self.surface.blit(self._static, (0, 0))
        return self.surface

    def transform(self, points):
        return (np.asarray(points) - self.camera) * self.scale + np.array(self.size) / 2

    def sample(self, distance):
        t = self.track
        distance = distance % t.length if t.closed else np.clip(distance, 0, t.length)
        i = min(
            np.searchsorted(t.cumulative, distance, side="right") - 1, len(t.delta) - 1
        )
        tangent = t.delta[i] / t.lengths[i]
        p = t.segment_starts[i] + tangent * (distance - t.cumulative[i])
        return p, np.array([-tangent[1], tangent[0]])

    def stripe(self, surface, distance, finish=False):
        p, n = self.sample(distance)
        half = self.track.road_half_width - 5
        for i, offset in enumerate(np.arange(-half, half, 9)):
            a = self.transform(p + n * offset)
            b = self.transform(p + n * min(half, offset + 9))
            pygame.draw.line(
                surface,
                (231, 239, 240) if i % 2 else (21, 29, 36),
                a,
                b,
                max(2, round(8 * self.scale)),
            )
        label = "FINISH" if finish else "START"
        q = self.transform(p + n * (half + 16))
        surface.blit(self.font.render(label, True, (127, 233, 188)), q)

    def background(self):
        t = self.track
        surface = pygame.Surface(self.size)
        surface.fill(GROUND)
        low = self.camera - np.array(self.size) / (2 * self.scale)
        high = self.camera + np.array(self.size) / (2 * self.scale)
        for axis in (0, 1):
            for value in np.arange(math.floor(low[axis] / 40) * 40, high[axis], 40):
                a, b = low.copy(), high.copy()
                a[axis] = b[axis] = value
                pygame.draw.aaline(
                    surface, (25, 39, 48), self.transform(a), self.transform(b)
                )
        plus = t.points + self.normals * t.road_half_width
        minus = t.points - self.normals * t.road_half_width
        if t.closed:

            def area(p):
                return abs(
                    np.sum(
                        p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)
                    )
                )

            outer, inner = (plus, minus) if area(plus) > area(minus) else (minus, plus)
            polygon(surface, ROAD, self.transform(outer))
            polygon(surface, GROUND, self.transform(inner))
        else:
            polygon(surface, ROAD, self.transform(np.vstack((plus, minus[::-1]))))
            for end in (t.points[0], t.points[-1]):
                pygame.draw.circle(
                    surface,
                    ROAD,
                    self.transform(end),
                    round(t.road_half_width * self.scale),
                )
        for boundary in (plus, minus):
            pygame.draw.aalines(
                surface, (114, 139, 150), t.closed, self.transform(boundary)
            )
        for i, distance in enumerate(np.arange(0, t.length - 12, 18)):
            p, n = self.sample(distance)
            q, m = self.sample(distance + 11)
            for side in (-1, 1):
                color = (201, 118, 104) if i % 2 else (195, 207, 209)
                a = self.transform(p + n * side * (t.road_half_width - 3))
                b = self.transform(q + m * side * (t.road_half_width - 3))
                pygame.draw.line(surface, color, a, b, max(2, round(3 * self.scale)))
        for distance in np.arange(25, t.length - 20, 48):
            a, _ = self.sample(distance)
            b, _ = self.sample(distance + 16)
            pygame.draw.aaline(
                surface, (99, 116, 128), self.transform(a), self.transform(b)
            )
        for i, (x, y, w, h) in enumerate(t.obstacles):
            top = self.transform((x, y))
            rect = pygame.Rect(
                round(top[0]),
                round(top[1]),
                max(1, round(w * self.scale)),
                max(1, round(h * self.scale)),
            )
            if not surface.get_rect().colliderect(rect):
                continue
            pygame.draw.rect(surface, (8, 15, 21), rect.move(3, 5), border_radius=5)
            pygame.draw.rect(surface, (112, 71, 63), rect, border_radius=5)
            pygame.draw.rect(
                surface,
                (230, 158, 106),
                rect,
                max(1, round(self.scale)),
                border_radius=5,
            )
            if rect.width > 24 and rect.height > 18:
                surface.blit(
                    self.font.render(f"{i + 1:02d}", True, (245, 210, 171)),
                    (rect.x + 7, rect.y + 5),
                )
        self.stripe(surface, 0)
        if not t.closed:
            self.stripe(surface, t.length, finish=True)
        if t.name == "babft":
            for i, distance in enumerate(range(600, int(t.length), 600)):
                p, n = self.sample(distance)
                q = self.transform(p + n * (t.road_half_width + 24))
                if -30 < q[1] < self.size[1] + 30:
                    surface.blit(
                        self.font.render(f"SECTOR {i + 1:02d}", True, (101, 147, 163)),
                        q,
                    )
        return surface

    def minimap(self, target, rect, position):
        t = self.track
        pygame.draw.rect(target, (12, 21, 29), rect, border_radius=8)
        scale = min((rect.width - 20) / t.width, (rect.height - 20) / t.height)
        offset = np.array(rect.center) - np.array([t.width, t.height]) * scale / 2
        points = t.points * scale + offset
        pygame.draw.lines(
            target,
            (84, 114, 128),
            t.closed,
            points,
            max(2, round(2 * t.road_half_width * scale)),
        )
        for x, y, w, h in t.obstacles:
            p = np.array([x, y]) * scale + offset
            pygame.draw.rect(
                target, (223, 144, 97), (*p, max(2, w * scale), max(2, h * scale))
            )
        pygame.draw.circle(
            target, (108, 242, 187), np.asarray(position) * scale + offset, 5
        )
        if not t.closed:
            pygame.draw.circle(
                target, (243, 216, 135), t.points[-1] * scale + offset, 4
            )
        half = np.array(self.size) / (2 * self.scale)
        top = (self.camera - half) * scale + offset
        box = pygame.Rect(*top, *(2 * half * scale))
        previous = target.get_clip()
        target.set_clip(rect)
        pygame.draw.rect(target, (157, 185, 199), box, 1)
        target.set_clip(previous)
