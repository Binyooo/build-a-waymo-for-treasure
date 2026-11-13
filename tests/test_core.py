import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from boatbrain.config import Config
from boatbrain.gates import Circuit
from boatbrain.network import Network
from boatbrain.simulation import Fleet, batched_actions, rollout
from boatbrain.track import Track
from boatbrain.training import Evolution


class GateTests(unittest.TestCase):
    def test_count_all_eight_bit_patterns(self):
        c = Circuit()
        wires = [c.input() for _ in range(8)]
        counted = c.count(wires)
        for bits in itertools.product((0, 1), repeat=8):
            _, values = c.run(bits, trace=True)
            self.assertEqual(
                sum(values[n] << i for i, n in enumerate(counted)), sum(bits)
            )

    def test_signed_votes_zero_and_bias(self):
        for weights in itertools.product((-1, 0, 1), repeat=3):
            for bias in (-1, 0, 1):
                c = Circuit()
                pairs = [(c.input(), c.input()) for _ in range(3)]
                c.outputs = [c.neuron(pairs, weights, bias)]
                for values in itertools.product((-1, 0, 1), repeat=3):
                    bits = [bit for v in values for bit in (int(v > 0), int(v < 0))]
                    p, n = c.run(bits)[0]
                    expected = np.sign(
                        sum(w * v for w, v in zip(weights, values)) + bias
                    )
                    self.assertEqual(p - n, expected)
                    self.assertFalse(p and n)

    def test_network_gate_reference_batch_agreement(self):
        rng = np.random.default_rng(14)
        for hidden in (2, 4, 6):
            config = Config(hidden=hidden)
            networks = [Network.random(rng, config) for _ in range(16)]
            sensors = rng.integers(0, 2, (16, config.inputs), dtype=np.int8)
            actions = batched_actions(
                np.stack([n.genome for n in networks]), sensors, config
            )
            for n, x, y in zip(networks, sensors, actions):
                np.testing.assert_array_equal(n.action(x), n.reference(x))
                np.testing.assert_array_equal(y, n.reference(x))

    def test_all_valid_nested_sensor_patterns(self):
        n = Network.random(np.random.default_rng(23))
        for thresholds in itertools.product(range(6), repeat=3):
            bits = [int(i >= threshold) for threshold in thresholds for i in range(5)]
            np.testing.assert_array_equal(n.action(bits), n.reference(bits))

    def test_invalid_genomes_rejected(self):
        size = Config().parameters
        self.assertEqual(size, 74)
        for genome in ([0] * (size - 1), [2] * size, [0.5] * size):
            with self.assertRaises(ValueError):
                Network(genome)

    def test_serialization_and_export_netlist(self):
        n = Network.random(np.random.default_rng(3))
        with tempfile.TemporaryDirectory() as tmp:
            n.save(Path(tmp) / "brain.json")
            clone = Network.load(Path(tmp) / "brain.json")
            np.testing.assert_array_equal(n.genome, clone.genome)
            n.export(tmp)
            data = json.loads((Path(tmp) / "gates.json").read_text())
            x = np.arange(15) % 2
            values = []
            inputs = iter(x)
            for node in data["nodes"]:
                op = node["op"]
                wires = node["inputs"]
                if op == "CONST0":
                    value = 0
                elif op == "CONST1":
                    value = 1
                elif op == "INPUT":
                    value = int(next(inputs))
                elif op == "NOT":
                    value = values[wires[0]] ^ 1
                elif op == "AND":
                    value = values[wires[0]] & values[wires[1]]
                elif op == "OR":
                    value = values[wires[0]] | values[wires[1]]
                elif op == "XOR":
                    value = values[wires[0]] ^ values[wires[1]]
                else:
                    self.fail(op)
                values.append(value)
            action = [values[p] - values[m] for p, m in data["outputs"]]
            np.testing.assert_array_equal(action, n.action(x))


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track()

    def test_spawn_valid_and_boxes_nested(self):
        for fraction in (0, 0.2, 0.5, 0.8):
            f = Fleet(self.track, fraction=fraction)
            self.assertFalse(f.collision()[0])
            bits = f.sensors().reshape(3, 5)
            self.assertTrue((np.diff(bits, axis=1) >= 0).all())

    def test_finite_width_sensor_detects_off_axis_wall(self):
        track = Track()
        track.wall[:] = False
        origin = np.array([[402.0, 362.0]])
        # The cell overlaps the box width but misses its centerline.
        track.wall[91, 110] = True
        bits = track.sense(origin, np.array([0.0]))[0, 5:10]
        np.testing.assert_array_equal(bits, [0, 1, 1, 1, 1])
        self.assertTrue(track.occupied([[0, -1], [-1, 0], [1000, 0]]).all())

    def test_forward_reverse_and_fixed_servo(self):
        f = Fleet(self.track)
        f.step([[1, 1]])
        self.assertGreater(f.speed[0], 0)
        before = f.angle[0]
        f.step([[1, 1]])
        self.assertGreater(f.angle[0], before)
        reverse = Fleet(self.track)
        reverse.step([[0, -1]])
        self.assertLess(reverse.speed[0], 0)
        self.assertLess(reverse.progress[0], 0)

    def test_gate_and_accelerated_rollouts_match(self):
        track = self.track
        n = Network.random(np.random.default_rng(5))
        _, accelerated = rollout(track, n.genome[None, :], steps=80)
        gate = Fleet(track)
        for _ in range(80):
            if not gate.alive.any():
                break
            gate.step([n.action(gate.sensors()[0])])
        np.testing.assert_allclose(gate.position, accelerated.position)
        np.testing.assert_array_equal(gate.crashed, accelerated.crashed)


class TrainingTests(unittest.TestCase):
    def test_restart_is_exact(self):
        e = Evolution(population=8, seed=8)
        e.advance(steps=5)
        with tempfile.TemporaryDirectory() as tmp:
            e.save(tmp)
            resumed = Evolution.resume(Path(tmp) / "checkpoint.json")
            a, b = e.advance(steps=5), resumed.advance(steps=5)
            self.assertEqual(a, b)
            np.testing.assert_array_equal(e.population, resumed.population)
            self.assertTrue(np.isin(e.population, [-1, 0, 1]).all())
            with self.assertRaises(ValueError):
                resumed.advance(steps=6)


if __name__ == "__main__":
    unittest.main()
