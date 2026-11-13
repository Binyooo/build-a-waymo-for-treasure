import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import tempfile
import unittest
from pathlib import Path

import pygame

from boatbrain.ui import Lab


class UITests(unittest.TestCase):
    def test_controls_and_headless_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Lab(run_dir=tmp)
            self.assertEqual(lab.screen.get_size(), (1920, 1080))
            lab.activate("ai")
            self.assertFalse(lab.ai)
            lab.activate("pause")
            self.assertTrue(lab.paused)
            lab.activate("sensors")
            self.assertFalse(lab.sensors_visible)
            lab.activate("speed")
            self.assertEqual(lab.multiplier, 2)
            lab.activate("track")
            self.assertEqual(lab.track.name, "harbor")
            lab.activate("export")
            self.assertTrue((lab.run_dir / "wiring/gates.json").exists())
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
            screenshot = Path(tmp) / "screen.png"
            lab.run(frames=1, screenshot=str(screenshot))
            self.assertTrue(lab.ai)
            self.assertGreater(screenshot.stat().st_size, 10000)

    def test_resize_event_and_scaled_map_click(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Lab(run_dir=tmp)
            lab.world.zoom = 1.4
            lab.resize(1280, 800)
            self.assertEqual(lab.window.get_size(), (1280, 800))
            self.assertEqual(lab.world.zoom, 1.4)
            self.assertEqual(lab.present_rect.size, (1280, 720))
            self.assertEqual(lab.present_rect.y, 40)
            lab.activate("maps")
            lab.draw()
            center = lab.map_buttons["babft"].center
            physical = (
                round(center[0] * 1280 / 1920),
                round(center[1] * 720 / 1080) + 40,
            )
            pygame.event.post(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=physical)
            )
            pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=960, h=540))
            screenshot = Path(tmp) / "resized.png"
            lab.run(frames=1, screenshot=str(screenshot))
            self.assertEqual(lab.track.name, "babft")
            self.assertEqual(pygame.image.load(screenshot).get_size(), (960, 540))

    def test_resize_large_and_back_to_small(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Lab(run_dir=tmp)
            try:
                for size in ((2560, 1440), (1024, 768), (1920, 1080)):
                    lab.resize(*size)
                    lab.draw()
                    self.assertEqual(lab.window.get_size(), size)
                    self.assertGreater(lab.viewport.width, 0)
                    self.assertTrue(lab.screen.get_rect().contains(lab.viewport))
            finally:
                pygame.quit()

    def test_map_picker_selects_long_course(self):
        with tempfile.TemporaryDirectory() as tmp:
            lab = Lab(run_dir=tmp)
            lab.activate("maps")
            lab.draw()
            target = lab.map_buttons["babft"]
            pygame.event.post(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=target.center)
            )
            lab.run(frames=1)
            self.assertEqual(lab.track.name, "babft")
            self.assertFalse(lab.map_menu)
            self.assertFalse(lab.track.closed)
            self.assertEqual(lab.run_dir, Path(tmp) / "babft")
            self.assertGreater(lab.world.scale, 1)


if __name__ == "__main__":
    unittest.main()
