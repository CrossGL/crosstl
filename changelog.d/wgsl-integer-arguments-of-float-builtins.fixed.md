WGSL output converts an `int` or `uint` argument of a float built-in such as
`max(f, k)`, `step(k, f)`, `mix`, `pow`, `atan(y, x)`, `mod` or `fma` to `f32`,
as GLSL's overload resolution does, and replicates it beside a vector. Float-only
built-ins such as `sqrt(k)`, `sin(u)`, `length(iv)`, `normalize(iv)` and
`dot(v, iv)` convert it even when no argument is floating point. It used to pass
the integer as it was, which naga rejected (#4503).
