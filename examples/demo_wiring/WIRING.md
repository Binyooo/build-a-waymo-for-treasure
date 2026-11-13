# Wiring this controller

15 occupied bits -> 4 hidden ternary neurons -> 2 ternary outputs.
74 parameters; 40 nonzero parameters.
Constant-folded, shared and pruned netlist: 250 unary/two-input gates.
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
