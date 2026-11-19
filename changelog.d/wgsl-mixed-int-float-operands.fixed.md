WGSL output converts an `int` or `uint` operand of `+`, `-`, `*`, `/` and
the comparisons to `f32` when the other operand is a float, or to
`vecN<f32>` beside a float vector, as the C-style languages do implicitly.
It used to leave the integer as it was, which naga rejected (#2384).
