WGSL output converts an `int`, `uint`, `ivecN` or `uvecN` argument of a
floating-point built-in such as `max`, `clamp`, `mix` or `distance` to `f32` or
`vecN<f32>`, as GLSL's overload resolution does, and a mixed call takes the
float vector's type. It used to pass the integer argument as it was, which naga
rejected (#4503).
