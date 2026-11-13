import csv
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pygame

from .network import Network
from .rendering import WorldView, polygon
from .simulation import Fleet
from .track import TRACK_INFO, TRACKS, Track

BG = (12, 18, 26)
PANEL = (20, 29, 40)
TEXT = (226, 235, 240)
MUTED = (125, 147, 162)
MINT = (103, 235, 181)
BLUE = (94, 178, 242)
RED = (245, 113, 117)


class Lab:
    def __init__(self, model=None, track="oval", run_dir="runs/lab"):
        pygame.init()
        pygame.display.set_caption("BOATBRAIN / ternary driving lab")
        self.window = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
        self.screen = self.window
        self.present_rect = self.window.get_rect()
        self._layout_size = None
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Menlo,Consolas,monospace", 18)
        self.small = pygame.font.SysFont("Menlo,Consolas,monospace", 14)
        self.title = pygame.font.SysFont("Helvetica,Arial", 32, bold=True)
        self.large = pygame.font.SysFont("Menlo,Consolas,monospace", 42, bold=True)
        self.model_path = Path(model) if model else Path("models/demo.json")
        self.network = (
            Network.load(self.model_path)
            if self.model_path.exists()
            else Network.random(np.random.default_rng(7))
        )
        self.config = self.network.config
        self.track = Track(track, self.config)
        self.fleet = Fleet(self.track)
        self.ai, self.paused, self.sensors_visible = True, False, True
        self.multiplier, self.accumulator = 1, 0.0
        self.run_dir = Path(run_dir)
        self.process, self.log = None, None
        self.history = []
        self.last_poll, self.last_model_mtime = 0, 0
        self.status = (
            "Demo loaded / train this map for new layouts"
            if self.model_path.exists()
            else "Random brain / train to discover a driver"
        )
        self.buttons = {}
        self.trail = []
        self.map_menu = False
        self.map_buttons = {}
        self.base_run_dir = Path(run_dir)
        self._use_track_run_dir()
        self.resize(*self.window.get_size())

    def resize(self, width, height):
        """Scale the 1080p layout to fit smaller windows."""
        size = (max(1, int(width)), max(1, int(height)))
        if pygame.display.get_window_size() != size:
            self.window = pygame.display.set_mode(size, pygame.RESIZABLE)
        else:
            self.window = pygame.display.get_surface()
        layout = size if size[0] >= 1920 and size[1] >= 1080 else (1920, 1080)
        self.screen = self.window if layout == size else pygame.Surface(layout)
        scale = min(size[0] / layout[0], size[1] / layout[1])
        fitted = (max(1, round(layout[0] * scale)), max(1, round(layout[1] * scale)))
        self.present_rect = pygame.Rect((0, 0), fitted)
        self.present_rect.center = self.window.get_rect().center
        if layout != self._layout_size:
            zoom = self.world.zoom if hasattr(self, "world") else 1.0
            self._layout_size = layout
            self._background()
            self.world.zoom = zoom

    def ui_position(self, position):
        """Map window coordinates to UI coordinates."""
        x, y = position
        rect = self.present_rect
        return (
            (x - rect.x) * self.screen.get_width() / rect.width,
            (y - rect.y) * self.screen.get_height() / rect.height,
        )

    def _use_track_run_dir(self):
        self.run_dir = self.base_run_dir / self.track.name
        self.history = []
        self.last_model_mtime = 0
        self.last_poll = 0

    def _background(self):
        width, height = self.screen.get_size()
        self.viewport = pygame.Rect(24, 144, width - 472, height - 176)
        self.panel_x = width - 424
        self.world = WorldView(self.track, self.viewport.size)

    def select_track(self, name):
        if self.process and self.process.poll() is None:
            self.status = "Stop the current trainer before selecting another map"
            self.map_menu = False
            return
        self.track = Track(name, self.config)
        self._use_track_run_dir()
        self._background()
        self.reset()
        self.map_menu = False
        self.status = self.track.description

    def text(self, text, xy, color=TEXT, font=None, surface=None):
        (surface or self.screen).blit(
            (font or self.font).render(str(text), True, color), xy
        )

    def button(self, name, label, rect, active=False):
        rect = pygame.Rect(rect)
        self.buttons[name] = rect
        hover = rect.collidepoint(self.ui_position(pygame.mouse.get_pos()))
        pygame.draw.rect(
            self.screen,
            (38, 65, 62) if active else ((43, 57, 72) if hover else PANEL),
            rect,
            border_radius=8,
        )
        pygame.draw.rect(
            self.screen, MINT if active else (52, 69, 82), rect, 1, border_radius=8
        )
        label_surface = self.small.render(label, True, MINT if active else TEXT)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def reset(self):
        self.fleet = Fleet(self.track)
        self.trail.clear()
        self.accumulator = 0

    def training(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.status = "Stopping trainer; saving checkpoint..."
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.log:
            self.log.close()
        self.log = (self.run_dir / "trainer.log").open("a")
        cmd = [
            sys.executable,
            "-m",
            "boatbrain",
            "train",
            "--out",
            str(self.run_dir),
            "--generations",
            "100",
        ]
        cmd += ["--tracks", self.track.name]
        checkpoint = self.run_dir / "checkpoint.json"
        if checkpoint.exists():
            import json

            steps = json.loads(checkpoint.read_text()).get("evaluation_steps") or 900
            cmd += ["--resume", str(checkpoint), "--steps", str(steps)]
        else:
            self.network.save(self.run_dir / "warm.json")
            cmd += [
                "--warm",
                str(self.run_dir / "warm.json"),
                "--steps",
                "3000" if not self.track.closed else "900",
            ]
        self.process = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT)
        self.status = "Training in background / first generation pending"

    def activate(self, action):
        if action == "ai":
            self.ai = not self.ai
        elif action == "pause":
            self.paused = not self.paused
        elif action == "reset":
            self.reset()
        elif action == "train":
            self.training()
        elif action == "sensors":
            self.sensors_visible = not self.sensors_visible
        elif action == "speed":
            self.multiplier = {1: 2, 2: 4, 4: 1}[self.multiplier]
        elif action == "track":
            self.select_track(TRACKS[(TRACKS.index(self.track.name) + 1) % len(TRACKS)])
        elif action == "maps":
            self.map_menu = not self.map_menu
        elif action == "export":
            count = self.network.export(self.run_dir / "wiring")
            self.status = f"Exported wiring / {count} gates"

    def poll(self):
        now = time.monotonic()
        if now - self.last_poll < 1:
            return
        self.last_poll = now
        path = self.run_dir / "best.json"
        if path.exists() and path.stat().st_mtime_ns != self.last_model_mtime:
            try:
                network = Network.load(path)
                if network.config != self.config:
                    self.status = "Checkpoint config differs; use another --run-dir"
                    return
                self.network = network
                self.last_model_mtime = path.stat().st_mtime_ns
                self.model_path = path
                self.status = "Latest champion loaded / Boolean inference"
                self.reset()
            except (ValueError, OSError):
                pass
        try:
            with (self.run_dir / "history.csv").open() as f:
                self.history = list(csv.DictReader(f))
        except OSError:
            pass
        if self.process and self.process.poll() is not None:
            if self.process.returncode not in (0, -15):
                self.status = "Trainer failed / see trainer.log"

    def draw_map_menu(self):
        s = self.screen
        shade = pygame.Surface(s.get_size(), pygame.SRCALPHA)
        shade.fill((3, 8, 15, 205))
        s.blit(shade, (0, 0))
        left = (s.get_width() - 1160) // 2
        panel = pygame.Rect(left, 170, 1160, 704)
        pygame.draw.rect(s, PANEL, panel, border_radius=16)
        pygame.draw.rect(s, (75, 100, 119), panel, 1, border_radius=16)
        self.text("SELECT A COURSE", (left + 32, 197), TEXT, self.title)
        self.text(
            "8 maps / circuits and point-to-point courses", (left + 32, 245), MUTED
        )
        self.map_buttons.clear()
        for i, name in enumerate(TRACKS):
            rect = pygame.Rect(
                left + 30 + (i % 2) * 555, 292 + (i // 2) * 132, 535, 112
            )
            self.map_buttons[name] = rect
            selected = self.track.name == name
            hover = rect.collidepoint(self.ui_position(pygame.mouse.get_pos()))
            pygame.draw.rect(
                s,
                (32, 65, 61) if selected else ((37, 53, 68) if hover else BG),
                rect,
                border_radius=9,
            )
            pygame.draw.rect(
                s, MINT if selected else (54, 77, 95), rect, 1, border_radius=9
            )
            title, description = TRACK_INFO[name]
            self.text(
                f"{i + 1:02d} / {title.upper()}",
                (rect.x + 18, rect.y + 16),
                MINT if selected else TEXT,
            )
            self.text(description, (rect.x + 18, rect.y + 48), MUTED, self.small)
            kind = "POINT TO POINT" if name in ("babft", "switchback") else "CIRCUIT"
            self.text(kind, (rect.x + 18, rect.y + 78), BLUE, self.small)
        self.text("Click a map / ESC to close", (left + 32, 839), MUTED, self.small)

    def draw(self):
        s = self.screen
        s.fill(BG)
        self.text("BOATBRAIN", (24, 22), MINT, self.title)
        self.text("TERNARY AUTONOMY LAB / 1080p", (280, 36), MUTED, self.small)
        self.text(
            "BUILD A BOAT / DRIVING SANDBOX", (self.panel_x, 36), MUTED, self.small
        )
        buttons = [
            ("ai", "AI DRIVER" if self.ai else "MANUAL", 140, self.ai),
            (
                "train",
                "STOP TRAIN"
                if self.process and self.process.poll() is None
                else "TRAIN THIS MAP",
                180,
                False,
            ),
            ("pause", "RESUME" if self.paused else "PAUSE", 120, self.paused),
            ("reset", "RESET [R]", 132, False),
            ("maps", "MAP / " + self.track.title.upper(), 240, self.map_menu),
            ("sensors", "SENSOR BOXES [L]", 185, self.sensors_visible),
            ("speed", f"{self.multiplier}x SPEED", 128, False),
            ("export", "EXPORT WIRING", 180, False),
        ]
        x = 24
        for key, label, width, active in buttons:
            self.button(key, label, (x, 86, width, 40), active)
            x += width + 10
        pos, angle = self.fleet.position[0], self.fleet.angle[0]
        bits = self.fleet.sensors()[0]
        view = self.world.frame(pos, angle)
        if len(self.trail) > 1:
            pygame.draw.aalines(
                view, (52, 153, 120), False, self.world.transform(self.trail)
            )
        if self.sensors_visible:
            layer = pygame.Surface(view.get_size(), pygame.SRCALPHA)
            for b, relative in enumerate(self.config.sensor_angles):
                for j in reversed(range(len(self.config.sensor_lengths))):
                    length = self.config.sensor_lengths[j]
                    points = self.world.transform(
                        self.track.sensor_polygon(pos, angle, relative, length)
                    )
                    hit = bits[b * len(self.config.sensor_lengths) + j]
                    pygame.draw.polygon(
                        layer, (*RED, 32) if hit else (*MINT, 18), points
                    )
                    pygame.draw.aalines(
                        layer, (*RED, 185) if hit else (*MINT, 170), True, points
                    )
            view.blit(layer, (0, 0))
        corners = self.world.transform(self.fleet.corners()[0])
        polygon(view, (5, 13, 18), corners + np.array([3, 5]))
        color = MINT if self.fleet.alive[0] else RED
        polygon(view, color, corners)
        axis = np.array([math.cos(angle), math.sin(angle)])
        normal = np.array([-axis[1], axis[0]])
        windshield = [
            pos + axis * a + normal * b for a, b in [(2, -5), (8, -5), (8, 5), (2, 5)]
        ]
        polygon(view, (17, 46, 54), self.world.transform(windshield))
        steering_angle = (
            angle + math.radians(self.config.servo_degrees) * self.fleet.actions[0, 0]
        )
        wheel_axis = np.array([math.cos(steering_angle), math.sin(steering_angle)])
        for side in (-1, 1):
            center = pos + axis * 8 + normal * side * 8
            pygame.draw.line(
                view,
                (7, 14, 19),
                self.world.transform(center - wheel_axis * 4),
                self.world.transform(center + wheel_axis * 4),
                max(2, round(3 * self.world.scale)),
            )
        s.blit(view, self.viewport.topleft)
        pygame.draw.rect(s, (58, 83, 101), self.viewport, 1, border_radius=8)
        badge = pygame.Rect(
            self.viewport.x + 20,
            self.viewport.y + 18,
            min(740, self.viewport.width - 40),
            84,
        )
        pygame.draw.rect(s, (13, 23, 32), badge, border_radius=7)
        self.text(
            self.track.title.upper()
            + (" / CIRCUIT" if self.track.closed else " / POINT TO POINT"),
            (badge.x + 15, badge.y + 12),
            TEXT,
        )
        self.text(self.status[:76], (badge.x + 15, badge.y + 44), MUTED, self.small)
        px = self.panel_x
        pygame.draw.rect(
            s, PANEL, (px, 144, 400, s.get_height() - 176), border_radius=10
        )
        self.text("LIVE TELEMETRY", (px + 24, 166), MUTED, self.small)
        progress = self.fleet.progress[0] / self.track.length
        self.text(
            f"{progress:.2f}"
            if self.track.closed
            else f"{max(0, progress) * 100:.1f}%",
            (px + 24, 196),
            TEXT,
            self.large,
        )
        self.text(
            "DIRECTED LAPS" if self.track.closed else "COURSE COMPLETE",
            (px + 190, 222),
            MUTED,
            self.small,
        )
        self.text(
            f"{abs(self.fleet.speed[0]):5.1f} px/s  [display only]",
            (px + 24, 255),
            MUTED,
            self.small,
        )
        status = (
            "PAUSED"
            if self.paused
            else (
                "RUNNING"
                if self.fleet.alive[0]
                else (
                    "CRASHED / R TO RESET"
                    if self.fleet.crashed[0]
                    else (
                        "FINISHED / R TO RESET"
                        if self.fleet.finished[0]
                        else "STALLED / R TO RESET"
                    )
                )
            )
        )
        self.text(
            status,
            (px + 24, 285),
            MINT if not self.fleet.crashed[0] else RED,
            self.small,
        )
        self.text("BOUNDING-BOX SENSORS", (px + 24, 330), MUTED, self.small)
        for beam, label in enumerate(("LEFT -55", "FRONT 0", "RIGHT +55")):
            self.text(label, (px + 24, 369 + beam * 34), TEXT, self.small)
            for j, length in enumerate(self.config.sensor_lengths):
                rect = pygame.Rect(px + 142 + j * 46, 363 + beam * 34, 39, 27)
                hit = bits[beam * 5 + j]
                pygame.draw.rect(s, RED if hit else (39, 83, 69), rect, border_radius=4)
                label_s = self.small.render(str(length), True, BG if hit else MINT)
                s.blit(label_s, label_s.get_rect(center=rect.center))
        self.text("red: occupied / green: clear", (px + 24, 479), MUTED, self.small)
        self.text(
            f"TERNARY MLP   15 > {self.config.hidden} > 2",
            (px + 24, 521),
            MUTED,
            self.small,
        )
        _, values = self.network.circuit.run(bits, trace=True)
        hidden = [values[p] - values[n] for p, n in self.network.circuit.hidden]
        for i, value in enumerate(hidden):
            cx = px + 48 + i * min(88, 304 / max(1, len(hidden) - 1))
            pygame.draw.circle(
                s,
                MINT if value > 0 else (RED if value < 0 else MUTED),
                (round(cx), 575),
                20,
                2,
            )
            self.text(f"{value:+d}", (cx - 12, 566), TEXT, self.small)
        steer, drive = self.fleet.actions[0]
        self.text(
            "STEER " + {-1: "LEFT", 0: "CENTER", 1: "RIGHT"}[int(steer)],
            (px + 24, 615),
            BLUE,
            self.small,
        )
        self.text(
            "DRIVE " + {-1: "REVERSE", 0: "COAST", 1: "FORWARD"}[int(drive)],
            (px + 210, 615),
            MINT,
            self.small,
        )
        self.text(
            f"{self.config.parameters} params / {np.count_nonzero(self.network.genome)} nonzero",
            (px + 24, 645),
            MUTED,
            self.small,
        )
        if not self.track.closed:
            self.text("COURSE OVERVIEW", (px + 24, 693), MUTED, self.small)
            self.world.minimap(s, pygame.Rect(px + 24, 720, 135, 204), pos)
            self.text(f"{self.track.length:,.0f} px", (px + 180, 740), TEXT)
            self.text("total length", (px + 180, 772), MUTED, self.small)
            self.text(
                f"{len(self.track.obstacles)} obstacles",
                (px + 180, 812),
                TEXT,
                self.small,
            )
            self.text("Wheel / +/-: zoom", (px + 180, 860), MUTED, self.small)
        else:
            self.text("EVOLUTION / CHAMPION FITNESS", (px + 24, 693), MUTED, self.small)
            chart = pygame.Rect(px + 24, 726, 352, 158)
            pygame.draw.rect(s, BG, chart, border_radius=7)
            if len(self.history) > 1:
                values_h = [float(r["champion"]) for r in self.history]
                low, high = min(0, min(values_h)), max(0.1, max(values_h))
                points = [
                    (
                        chart.left + i / (len(values_h) - 1) * chart.width,
                        chart.bottom
                        - 12
                        - (v - low) / (high - low) * (chart.height - 24),
                    )
                    for i, v in enumerate(values_h)
                ]
                pygame.draw.aalines(s, MINT, False, points)
                self.text(
                    f"GEN {self.history[-1]['generation']} / BEST {values_h[-1]:.3f}",
                    (px + 24, 905),
                    MINT,
                    self.small,
                )
            else:
                self.text("TRAIN THIS MAP to evolve", (px + 43, 791), MUTED, self.small)
        for i, line in enumerate(
            (
                "WASD / arrows: manual drive",
                "TAB: AI/manual   SPACE: pause",
                "R: reset  L: boxes  M: maps",
            )
        ):
            self.text(line, (px + 24, s.get_height() - 124 + i * 27), MUTED, self.small)
        self.text(
            "LIDAR ONLY / 30 DEGREE SERVO / BOOLEAN INFERENCE",
            (self.viewport.x + 20, self.viewport.bottom - 30),
            MUTED,
            self.small,
        )
        if self.map_menu:
            self.draw_map_menu()
        if self.screen is not self.window:
            self.window.fill(BG)
            scaled = pygame.transform.smoothscale(self.screen, self.present_rect.size)
            self.window.blit(scaled, self.present_rect)
        pygame.display.flip()

    def run(self, frames=None, screenshot=None):
        running, counter = True, 0
        try:
            while running:
                elapsed = min(0.1, self.clock.tick(60) / 1000)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self.resize(event.w, event.h)
                    elif event.type in (pygame.WINDOWRESIZED, pygame.WINDOWSIZECHANGED):
                        self.resize(*pygame.display.get_window_size())
                    elif event.type == pygame.MOUSEWHEEL:
                        self.world.zoom = float(
                            np.clip(self.world.zoom * (1.1**event.y), 0.65, 2.5)
                        )
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if not self.present_rect.collidepoint(event.pos):
                            continue
                        position = self.ui_position(event.pos)
                        if self.map_menu:
                            for name, rect in self.map_buttons.items():
                                if rect.collidepoint(position):
                                    self.select_track(name)
                                    break
                            else:
                                self.map_menu = False
                        else:
                            for name, rect in self.buttons.items():
                                if rect.collidepoint(position):
                                    self.activate(name)
                                    break
                    elif event.type == pygame.KEYDOWN:
                        keys = {
                            pygame.K_TAB: "ai",
                            pygame.K_SPACE: "pause",
                            pygame.K_r: "reset",
                            pygame.K_l: "sensors",
                            pygame.K_t: "train",
                            pygame.K_m: "maps",
                        }
                        if event.key in keys:
                            self.activate(keys[event.key])
                        elif event.key in (
                            pygame.K_EQUALS,
                            pygame.K_PLUS,
                            pygame.K_MINUS,
                        ):
                            self.world.zoom = float(
                                np.clip(
                                    self.world.zoom
                                    * (1.1 if event.key != pygame.K_MINUS else 1 / 1.1),
                                    0.65,
                                    2.5,
                                )
                            )
                        elif event.key == pygame.K_ESCAPE:
                            if self.map_menu:
                                self.map_menu = False
                            else:
                                running = False
                self.poll()
                if not self.paused and not self.map_menu:
                    self.accumulator += elapsed * self.multiplier
                    while self.accumulator >= self.config.dt:
                        bits = self.fleet.sensors()[0]
                        if self.ai:
                            action = self.network.action(bits)
                        else:
                            keys = pygame.key.get_pressed()
                            action = (
                                int(keys[pygame.K_d] or keys[pygame.K_RIGHT])
                                - int(keys[pygame.K_a] or keys[pygame.K_LEFT]),
                                int(keys[pygame.K_w] or keys[pygame.K_UP])
                                - int(keys[pygame.K_s] or keys[pygame.K_DOWN]),
                            )
                        self.fleet.step([action])
                        self.accumulator -= self.config.dt
                        self.trail.append(self.fleet.position[0].copy())
                        self.trail = self.trail[-500:]
                self.draw()
                counter += 1
                if frames and counter >= frames:
                    running = False
            if screenshot:
                pygame.image.save(self.window, screenshot)
        finally:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.log:
                self.log.close()
            pygame.quit()
