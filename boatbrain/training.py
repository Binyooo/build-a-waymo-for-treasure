import csv
import json
from pathlib import Path

import numpy as np

from .config import Config
from .network import Network
from .simulation import rollout
from .track import Track


class Evolution:
    def __init__(
        self,
        population=64,
        seed=7,
        config=Config(),
        warm=None,
        tracks=("oval", "harbor"),
    ):
        if population < 8:
            raise ValueError("Population must be at least 8")
        self.config, self.rng = config, np.random.default_rng(seed)
        self.population = self.rng.choice(
            [-1, 0, 1], (population, config.parameters), p=[0.22, 0.56, 0.22]
        ).astype(np.int8)
        h, n = config.hidden, config.inputs
        self.population[:, n * h + h + h : n * h + h + 2 * h] = 0
        self.population[:, -1] = 1  # Start with forward drive; mutations can change it.
        if warm is not None:
            self.population[0] = warm.genome
        self.generation, self.best, self.best_score = 0, None, -float("inf")
        self.history = []
        self.evaluation_steps = None
        if not tracks or len(set(tracks)) != len(tracks):
            raise ValueError("Choose at least one distinct training track")
        self.track_names = tuple(tracks)
        self.tracks = [Track(name, config) for name in tracks]

    def advance(self, steps=900):
        if self.evaluation_steps is not None and self.evaluation_steps != steps:
            raise ValueError(
                "Use the same --steps as the checkpoint so fitness stays comparable"
            )
        self.evaluation_steps = steps
        scores = []
        scenarios = [
            (track, fraction, direction, offset)
            for track in self.tracks
            for fraction, direction, offset in track.scenarios()
        ]
        for track, fraction, direction, offset in scenarios:
            score, _ = rollout(
                track, self.population, steps, fraction, direction, offset
            )
            scores.append(score)
        scores = np.stack(scores)
        fitness = 0.65 * scores.mean(axis=0) + 0.35 * scores.min(axis=0)
        fitness -= (
            0.003 * np.count_nonzero(self.population, axis=1) / self.config.parameters
        )
        order = np.argsort(fitness)[::-1]
        champion = int(order[0])
        if fitness[champion] > self.best_score:
            self.best_score = float(fitness[champion])
            self.best = Network(self.population[champion].copy(), self.config)
        row = {
            "generation": self.generation,
            "best": float(fitness[champion]),
            "mean": float(fitness.mean()),
            "champion": self.best_score,
            "nonzero": int(np.count_nonzero(self.best.genome)),
        }
        for i, name in enumerate(self.track_names):
            row[f"{name}_reward"] = float(scores[3 * i : 3 * i + 3, champion].mean())
        self.history.append(row)
        elite_count = max(2, len(order) // 8)
        new = [g.copy() for g in self.population[order[:elite_count]]]
        new[0] = self.best.genome.copy()
        while len(new) < len(order):
            if self.rng.random() < 0.07:
                child = self.rng.choice(
                    [-1, 0, 1], self.config.parameters, p=[0.2, 0.6, 0.2]
                ).astype(np.int8)
            else:

                def tournament():
                    candidates = self.rng.integers(len(order), size=4)
                    return self.population[candidates[np.argmax(fitness[candidates])]]

                a, b = tournament(), tournament()
                child = (
                    np.where(self.rng.random(self.config.parameters) < 0.5, a, b).copy()
                    if self.rng.random() < 0.3
                    else a.copy()
                )
                rate = self.rng.choice([0.015, 0.04, 0.10], p=[0.45, 0.4, 0.15])
                mask = self.rng.random(self.config.parameters) < rate
                mask[self.rng.integers(self.config.parameters)] = True
                child[mask] = (
                    (child[mask] + 1 + self.rng.integers(1, 3, mask.sum())) % 3
                ) - 1
            new.append(child)
        self.population = np.asarray(new, dtype=np.int8)
        self.generation += 1
        return row

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.best:
            self.best.save(
                directory / "best.json",
                {"fitness": self.best_score, "generation": self.generation},
            )
        if self.history:
            with (directory / "history.tmp").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
                writer.writeheader()
                writer.writerows(self.history)
            (directory / "history.tmp").replace(directory / "history.csv")
        checkpoint = {
            "scenario_version": 2,
            "tracks": self.track_names,
            "config": self.config.to_dict(),
            "generation": self.generation,
            "population": self.population.tolist(),
            "rng": self.rng.bit_generator.state,
            "evaluation_steps": self.evaluation_steps,
            "best": self.best.genome.tolist() if self.best else None,
            "best_score": self.best_score if self.best else None,
            "history": self.history,
        }
        temp = directory / "checkpoint.tmp"
        temp.write_text(json.dumps(checkpoint))
        temp.replace(directory / "checkpoint.json")

    @classmethod
    def resume(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("scenario_version") != 2:
            raise ValueError(
                "Checkpoint uses older training scenarios; use --warm with its best.json instead"
            )
        obj = cls(
            len(data["population"]),
            config=Config.from_dict(data["config"]),
            tracks=data.get("tracks", ("oval", "harbor")),
        )
        obj.population = np.array(data["population"], dtype=np.int8)
        obj.rng.bit_generator.state = data["rng"]
        obj.generation, obj.history = data["generation"], data["history"]
        obj.evaluation_steps = data.get("evaluation_steps")
        obj.best_score = (
            data["best_score"] if data["best_score"] is not None else -float("inf")
        )
        obj.best = (
            Network(data["best"], obj.config) if data["best"] is not None else None
        )
        return obj


def evaluate(network, steps=1800, tracks=("oval", "harbor", "technical")):
    results = []
    for name in tracks:
        track = Track(name, network.config)
        for fraction, direction, offset in track.scenarios():
            _, fleet = rollout(
                track, network.genome[None, :], steps, fraction, direction, offset
            )
            results.append(
                {
                    "track": name,
                    "course_type": "circuit" if track.closed else "point_to_point",
                    "finished": bool(fleet.finished[0]),
                    "start": fraction,
                    "direction": direction,
                    "offset": offset,
                    "laps": round(float(fleet.progress[0] / track.length), 4),
                    "crashed": bool(fleet.crashed[0]),
                    "steps": int(fleet.steps[0]),
                }
            )
    return results
