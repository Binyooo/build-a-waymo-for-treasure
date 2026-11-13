"""Boolean circuits with dual-rail signals: +1=(1,0), 0=(0,0), -1=(0,1)."""

from dataclasses import dataclass


@dataclass
class Gate:
    op: str
    inputs: tuple[int, ...]


class Circuit:
    def __init__(self):
        self.nodes = [Gate("CONST0", ()), Gate("CONST1", ())]
        self.input_nodes = []
        self.outputs = []
        self._cache = {}

    def input(self):
        n = len(self.nodes)
        self.nodes.append(Gate("INPUT", ()))
        self.input_nodes.append(n)
        return n

    def gate(self, op, *args):
        if op == "NOT":
            if args[0] in (0, 1):
                return args[0] ^ 1
            node = self.nodes[args[0]]
            if node.op == "NOT":
                return node.inputs[0]
        else:
            args = tuple(sorted(args))
            a, b = args
            if op == "AND":
                if a == 0:
                    return 0
                if a == 1:
                    return b
                if a == b:
                    return a
            elif op == "OR":
                if a == 0:
                    return b
                if a == 1:
                    return 1
                if a == b:
                    return a
            elif op == "XOR":
                if a == 0:
                    return b
                if a == 1:
                    return self.gate("NOT", b)
                if a == b:
                    return 0
        key = (op, args)
        if key in self._cache:
            return self._cache[key]
        n = len(self.nodes)
        self.nodes.append(Gate(op, args))
        self._cache[key] = n
        return n

    def count(self, wires):
        width = max(1, len(wires).bit_length())
        acc = [0] * width
        for wire in wires:
            carry = wire
            for i in range(width):
                previous = acc[i]
                acc[i] = self.gate("XOR", previous, carry)
                carry = self.gate("AND", previous, carry)
        return acc

    def greater(self, a, b):
        equal, greater = 1, 0
        for x, y in zip(reversed(a), reversed(b)):
            x_not_y = self.gate("AND", x, self.gate("NOT", y))
            greater = self.gate("OR", greater, self.gate("AND", equal, x_not_y))
            equal = self.gate("AND", equal, self.gate("NOT", self.gate("XOR", x, y)))
        return greater

    def neuron(self, inputs, weights, bias):
        pos, neg = [], []
        for (p, n), w in zip(inputs, weights):
            if w == 1:
                pos.append(p)
                neg.append(n)
            elif w == -1:
                pos.append(n)
                neg.append(p)
        if bias:
            pos.append(1 if bias == 1 else 0)
            neg.append(1 if bias == -1 else 0)
        if not pos:
            return (0, 0)
        p, n = self.count(pos), self.count(neg)
        return self.greater(p, n), self.greater(n, p)

    def run(self, bits, trace=False):
        if len(bits) != len(self.input_nodes):
            raise ValueError("Wrong number of sensor bits")
        if any(b not in (0, 1) for b in bits):
            raise ValueError("Sensors must be Boolean")
        values = [0, 1]
        iterator = iter(bits)
        for node in self.nodes[2:]:
            if node.op == "INPUT":
                value = int(next(iterator))
            elif node.op == "NOT":
                value = values[node.inputs[0]] ^ 1
            else:
                a, b = (values[i] for i in node.inputs)
                if node.op == "AND":
                    value = a & b
                elif node.op == "OR":
                    value = a | b
                elif node.op == "XOR":
                    value = a ^ b
                else:
                    raise ValueError(node.op)
            values.append(value)
        rails = [(values[p], values[n]) for p, n in self.outputs]
        return (rails, values) if trace else rails

    def to_dict(self):
        # Keep input pins even when no output uses them.
        reachable = {0, 1, *self.input_nodes}
        stack = [n for pair in self.outputs for n in pair]
        while stack:
            node = stack.pop()
            if node in reachable:
                continue
            reachable.add(node)
            stack.extend(self.nodes[node].inputs)
        ordered = sorted(reachable)
        remap = {old: new for new, old in enumerate(ordered)}
        return {
            "encoding": "+1=(1,0), 0=(0,0), -1=(0,1)",
            "input_nodes": [remap[n] for n in self.input_nodes],
            "outputs": [(remap[p], remap[n]) for p, n in self.outputs],
            "nodes": [
                {
                    "id": remap[i],
                    "op": self.nodes[i].op,
                    "inputs": [remap[n] for n in self.nodes[i].inputs],
                }
                for i in ordered
            ],
        }
