GLSL output from CrossGL source computes 8- and 16-bit integer scalar arithmetic
on `int`, as CrossGL defines it: `int8_t v = 13; float(v * v)` is `169.0`, not
the 8-bit `-87.0`. Only an explicit conversion back, `int8_t(v * v)`, wraps.
GLSL input writes GLSL's own narrow results explicitly in CrossGL
(`int8_t((uint(v) * uint(v)))`), so GLSL that goes through CrossGL source keeps
computing at the operands' width. GLSL that mixes two narrow integer types in one
operation is refused with `glsl-narrow-arithmetic-untyped`.
