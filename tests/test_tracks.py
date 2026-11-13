import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from boatbrain.simulation import Fleet
from boatbrain.track import TRACKS, Track
from boatbrain.training import Evolution


class NewMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps = {name: Track(name) for name in TRACKS[3:]}

    def test_five_new_maps_have_safe_starts_and_nested_sensors(self):
        self.assertEqual(len(self.maps), 5)
        for name, track in self.maps.items():
            with self.subTest(track=name):
                for fraction, direction, offset in track.scenarios():
                    fleet = Fleet(
                        track, fraction=fraction, direction=direction, offset=offset
                    )
                    self.assertFalse(fleet.collision()[0])
                    self.assertTrue(
                        (np.diff(fleet.sensors().reshape(3, 5), axis=1) >= 0).all()
                    )

    def test_obstacles_are_shared_by_lidar_and_collision(self):
        for name in ("babft", "slalom"):
            track = self.maps[name]
            self.assertGreater(len(track.obstacles), 5)
            x, y, w, h = track.obstacles[0]
            fleet = Fleet(track)
            fleet.position[0] = [x + w / 2, y + h / 2]
            self.assertTrue(fleet.collision()[0])
            self.assertTrue(fleet.sensors().all())

    def test_long_course_has_a_car_sized_path_to_finish(self):
        track = self.maps["babft"]
        self.assertGreater(track.length, 5000)
        self.assertFalse(track.closed)
        # Reserve 16px clearance around the route.
        base = ~track.wall[::2, ::2]
        free = base.copy()
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                free &= np.roll(base, (dy, dx), axis=(0, 1))
        start = tuple((track.points[0][::-1] / 8).astype(int))
        finish = tuple((track.points[-1][::-1] / 8).astype(int))
        queue = deque([start])
        visited = {start}
        while queue:
            y, x = queue.popleft()
            if (y, x) == finish:
                break
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (y + dy, x + dx)
                if (
                    0 <= nxt[0] < free.shape[0]
                    and 0 <= nxt[1] < free.shape[1]
                    and free[nxt]
                    and nxt not in visited
                ):
                    visited.add(nxt)
                    queue.append(nxt)
        self.assertIn(finish, visited)

    def test_open_course_finish_both_directions_without_wrapping(self):
        track = self.maps["babft"]
        for fraction, direction in ((0.99, 1), (0.01, -1)):
            fleet = Fleet(track, fraction=fraction, direction=direction)
            for _ in range(30):
                if not fleet.alive[0]:
                    break
                fleet.speed[0] = 90
                fleet.step([[0, 1]])
            self.assertTrue(fleet.finished[0])
            self.assertFalse(fleet.crashed[0])
            self.assertLess(fleet.progress[0], 100)
            self.assertGreater(fleet.progress[0], 30)
        start = Fleet(track)
        start.step([[0, -1]])
        self.assertFalse(start.finished[0])
        self.assertLessEqual(start.progress[0], 0)

    def test_custom_training_maps_survive_checkpoint(self):
        evolution = Evolution(population=8, seed=4, tracks=("babft",))
        evolution.advance(steps=3)
        with tempfile.TemporaryDirectory() as directory:
            evolution.save(directory)
            resumed = Evolution.resume(Path(directory) / "checkpoint.json")
            self.assertEqual(resumed.track_names, ("babft",))
            self.assertEqual(evolution.advance(3), resumed.advance(3))
            np.testing.assert_array_equal(evolution.population, resumed.population)


if __name__ == "__main__":
    unittest.main()
