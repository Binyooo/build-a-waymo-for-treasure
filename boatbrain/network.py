import json
from pathlib import Path

import numpy as np

from .config import Config
from .gates import Circuit


class Network:
    def __init__(self, genome, config=Config()):
        self.config = config
        raw = np.asarray(genome)
        if raw.shape != (config.parameters,) or not np.isin(raw, [-1, 0, 1]).all():
            raise ValueError(f"Expected {config.parameters} ternary parameters")
        self.genome = raw.astype(np.int8)
        k, h = config.inputs * config.hidden, config.hidden
        self.w1 = self.genome[:k].reshape(h, config.inputs)
        self.b1 = self.genome[k : k + h]
        self.w2 = self.genome[k + h : k + h + 2 * h].reshape(2, h)
        self.b2 = self.genome[-2:]
        self.circuit = self.compile()

    def compile(self):
        circuit, inputs = Circuit(), []
        for _ in range(self.config.inputs):
            occupied = circuit.input()
            inputs.append((circuit.gate("NOT", occupied), occupied))
        hidden = [circuit.neuron(inputs, w, int(b)) for w, b in zip(self.w1, self.b1)]
        circuit.hidden = hidden
        circuit.outputs = [
            circuit.neuron(hidden, w, int(b)) for w, b in zip(self.w2, self.b2)
        ]
        return circuit

    def action(self, occupied):
        # Decode output rails for the simulator.
        return tuple(p - n for p, n in self.circuit.run(occupied))

    def reference(self, occupied):
        x = 1 - 2 * np.asarray(occupied, dtype=np.int32)
        hidden = np.sign(x @ self.w1.T + self.b1)
        return np.sign(hidden @ self.w2.T + self.b2).astype(np.int8)

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": 1,
            "config": self.config.to_dict(),
            "genome": self.genome.tolist(),
            "metadata": metadata or {},
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("format") != 1:
            raise ValueError("Unsupported model format")
        return cls(data["genome"], Config.from_dict(data["config"]))

    @classmethod
    def random(cls, rng, config=Config()):
        return cls(
            rng.choice([-1, 0, 1], config.parameters, p=[0.22, 0.56, 0.22]), config
        )

    def export(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        netlist = self.circuit.to_dict()
        (directory / "gates.json").write_text(json.dumps(netlist, indent=2))
        gate_rows = ["id,operation,input_a,input_b"]
        for node in netlist["nodes"]:
            pins = list(node["inputs"]) + ["", ""]
            gate_rows.append(f"{node['id']},{node['op']},{pins[0]},{pins[1]}")
        (directory / "gates.csv").write_text("\n".join(gate_rows) + "\n")
        lines = ["layer,neuron,input,weight"]
        for layer, weights, biases in [
            ("hidden", self.w1, self.b1),
            ("output", self.w2, self.b2),
        ]:
            for i, (row, bias) in enumerate(zip(weights, biases)):
                lines += [f"{layer},{i},{j},{int(w)}" for j, w in enumerate(row)]
                lines.append(f"{layer},{i},bias,{int(bias)}")
        (directory / "weights.csv").write_text("\n".join(lines) + "\n")
        lines = ["index,direction_degrees,box_length,box_width"]
        for i, angle in enumerate(self.config.sensor_angles):
            for j, length in enumerate(self.config.sensor_lengths):
                lines.append(
                    f"{i * len(self.config.sensor_lengths) + j},{angle},{length},{self.config.sensor_width}"
                )
        (directory / "sensors.csv").write_text("\n".join(lines) + "\n")
        self.save(directory / "model.json")
        gates = sum(n["op"] in ("AND", "OR", "XOR", "NOT") for n in netlist["nodes"])
        text = f"""# Wiring this controller

{self.config.inputs} occupied bits -> {self.config.hidden} hidden ternary neurons -> 2 ternary outputs.
{self.config.parameters} parameters; {np.count_nonzero(self.genome)} nonzero parameters.
Constant-folded, shared and pruned netlist: {gates} unary/two-input gates.
Ternary weights do NOT imply one gate per weight. Zero weights omit connections.

Input order is sensors.csv: direction first, then short to long boxes.
Boxes start at the car center; an occupied bit is 1 if any wall intersects the box.
Inputs are (NOT occupied, occupied). Choose a uniform pixel-to-stud scale including
sensor lengths, car dimensions and wheelbase. Negative sensor angles point left.

weights.csv includes all weights and biases. For +1 preserve the two rails; for -1
swap them; for 0 omit the connection. A +1 bias adds one positive vote, -1 adds a
negative vote. A neuron outputs +1 if positive votes exceed negative votes, -1 if
fewer, or zero on a tie. Use combinational ripple counters and unsigned comparators.
Do not accumulate votes across frames. gates.json is the exact AND/OR/XOR/NOT DAG.
Nodes 0 and 1 are constants; evaluate ascending node IDs. INPUT nodes are ordered
by input_nodes. Outputs are pairs of node IDs, (positive, negative).

Output 0: positive RIGHT, negative LEFT, both low CENTER.
Output 1: positive FORWARD, negative BACKWARD, both low COAST.
Steering is +/-30 degrees. The controller receives no speed, position, heading,
progress or previous actions. Both rails high is never a valid ternary signal.

The sensor model rasterizes walls at 4 px, snaps origin to cell centers and
headings to 3 degrees, then intersects finite rectangles with occupied cells.
Real Roblox sensing, gate delays, fan-out and actuator lag need calibration.
"""
        (directory / "WIRING.md").write_text(text)
        return gates
