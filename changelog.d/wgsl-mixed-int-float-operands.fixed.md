WGSL output converts an `int` or `uint` scalar beside a float matrix to `f32`
(`m * i` becomes `m * f32(i)`). It used to leave the integer as it was, which
naga rejected (#2384).
