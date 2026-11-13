import math

import numpy as np

from .config import Config


class Fleet:
    """Fixed-step bicycle model."""

    def __init__(self, track, count=1, fraction=0, direction=1, offset=0):
        self.track, self.config = track, track.config
        pos, angle = track.spawn(fraction, direction, offset)
        self.position = np.tile(pos, (count, 1))
        self.angle = np.full(count, angle)
        self.speed = np.zeros(count)
        self.alive = np.ones(count, dtype=bool)
        self.progress = np.zeros(count)
        self.previous = track.project(self.position)[0]
        self.direction = direction
        self.finished = np.zeros(count, dtype=bool)
        self.steps = np.zeros(count, dtype=int)
        self.stalled = np.zeros(count, dtype=int)
        self.crashed = np.zeros(count, dtype=bool)
        self.actions = np.zeros((count, 2), dtype=np.int8)

    def sensors(self):
        return self.track.sense(self.position, self.angle)

    def step(self, actions):
        actions = np.asarray(actions)
        if (
            actions.shape != self.actions.shape
            or not np.isin(actions, [-1, 0, 1]).all()
        ):
            raise ValueError("Expected steering and drive in {-1,0,1}")
        c, alive = self.config, self.alive.copy()
        self.actions = actions.astype(np.int8)
        self.speed += (
            alive * (actions[:, 1] * c.acceleration - c.drag * self.speed) * c.dt
        )
        self.speed = np.clip(self.speed, -c.max_reverse, c.max_forward)
        for _ in range(2):
            moving = self.alive
            yaw = (
                self.speed
                / c.wheelbase
                * np.tan(actions[:, 0] * math.radians(c.servo_degrees))
            )
            self.angle += moving * yaw * c.dt / 2
            self.position += (
                moving[:, None]
                * self.speed[:, None]
                * np.column_stack((np.cos(self.angle), np.sin(self.angle)))
                * c.dt
                / 2
            )
            self.crashed |= moving & self.collision()
            self.alive &= ~self.crashed
        current = self.track.project(self.position)[0]
        delta = current - self.previous
        if self.track.closed:
            delta = (
                delta + self.track.length / 2
            ) % self.track.length - self.track.length / 2
        signed = delta * self.direction
        self.progress += alive * signed
        self.previous = current
        self.steps += alive
        self.stalled = np.where(alive & (signed < 0.05), self.stalled + 1, 0)
        if self.track.closed:
            self.finished |= self.progress >= 2 * self.track.length
        else:
            remaining = self.track.length - current if self.direction == 1 else current
            self.finished |= (remaining <= c.car_length / 2) & ~self.crashed
        self.alive &= (self.stalled < 150) & ~self.finished
        return self.alive

    def corners(self):
        c = self.config
        local = np.array(
            [
                [-c.car_length / 2, -c.car_width / 2],
                [c.car_length / 2, -c.car_width / 2],
                [c.car_length / 2, c.car_width / 2],
                [-c.car_length / 2, c.car_width / 2],
            ]
        )
        co, si = np.cos(self.angle), np.sin(self.angle)
        rotation = np.stack((co, -si, si, co), axis=1).reshape(-1, 2, 2)
        return self.position[:, None, :] + np.einsum("nij,kj->nki", rotation, local)

    def collision(self):
        corners = self.corners()
        samples = [self.position[:, None, :]]
        for t in (0, 0.25, 0.5, 0.75):
            samples.append(corners * (1 - t) + np.roll(corners, -1, axis=1) * t)
        return self.track.occupied(np.concatenate(samples, axis=1)).any(axis=1)


def batched_actions(genomes, occupied, config=Config()):
    """Evaluate a population using integer MLP inference."""
    n, h = config.inputs, config.hidden
    k = n * h
    w1 = genomes[:, :k].reshape(-1, h, n).astype(np.int32)
    b1 = genomes[:, k : k + h]
    w2 = genomes[:, k + h : k + h + 2 * h].reshape(-1, 2, h).astype(np.int32)
    x = 1 - 2 * occupied.astype(np.int32)
    hidden = np.sign(np.einsum("bhi,bi->bh", w1, x) + b1)
    return np.sign(np.einsum("boh,bh->bo", w2, hidden) + genomes[:, -2:]).astype(
        np.int8
    )


def rollout(track, genomes, steps=900, fraction=0, direction=1, offset=0):
    fleet = Fleet(track, len(genomes), fraction, direction, offset)
    for _ in range(steps):
        if not fleet.alive.any():
            break
        active = np.flatnonzero(fleet.alive)
        actions = np.zeros((len(genomes), 2), dtype=np.int8)
        occupied = track.sense(fleet.position[active], fleet.angle[active])
        actions[active] = batched_actions(genomes[active], occupied, track.config)
        fleet.step(actions)
    score = fleet.progress / track.length - fleet.crashed * 0.08
    return score, fleet
