SPIR-V input keeps 8- and 16-bit integer arithmetic at its width: a narrow
scalar `OpIAdd`, `OpISub`, `OpIMul`, `OpShiftLeftLogical`, `OpSNegate` or
unsigned `OpNot` imports as `i8(...)`, `u16(...)` and so on, because CrossGL
promotes narrow scalars to `int` before arithmetic. `OpBitcast` between 8-,
16- or 64-bit integers of one width now imports as a constructor instead of
being refused. The subgroup vote, ballot, shuffle, quad, rotate and partition
instructions now import as the GLSL subgroup functions (the quad votes as
`QuadAll` and `QuadAny`) instead of failing on an unknown value, and SPIR-V
output writes `QuadAll` and `QuadAny` with SPV_KHR_quad_control.
