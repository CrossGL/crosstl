HIP output writes arithmetic and built-in calls on a constant-buffer vector
member, as in `v - scale / 2.0` or `dot(scale, v)` for a `vec3 scale` in a
`cbuffer`, with the same `cgl_float3_*` helpers it uses for other vectors. It
used to treat the member as a scalar, so hipcc rejected the output or the
generator refused the program (#4722).
