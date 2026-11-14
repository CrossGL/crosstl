WGSL output converts an `int` or `ivecN` operand of `&`, `|` and `^`, and of
vector `min`, `max` and `clamp`, to the unsigned type when another operand is
`uint` or `uvecN`, as GLSL does. It used to leave the signed operand as it
was, which naga rejected (#4492).
