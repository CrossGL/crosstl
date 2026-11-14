CUDA output writes arithmetic on a constant-buffer vector member, as in
`offset + offset` or `offset * 2.0` for a `vec2 offset` in a `cbuffer`, with
the same `cgl_float2_add`-style helpers it uses for other vectors. It used to
write the C operator, which CUDA's `float2` does not define, so the output did
not compile (#4431).
