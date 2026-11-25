Mojo output knows that `abs`, `floor`, `sqrt`, `sin` and the other per-lane
math built-ins return the type of their operand, so `nextafter(abs(x), y)` and
`isinf(abs(x))` translate instead of being refused for an operand of unknown
type. A value named `_`, which SPIR-V input writes for a buffer without a
debug name, is renamed `__`, because a lone `_` is Mojo's discard pattern and
cannot name a kernel argument.
