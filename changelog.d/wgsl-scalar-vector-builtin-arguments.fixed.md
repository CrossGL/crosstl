WGSL output replicates a scalar argument of `min`, `max`, `step` and
`smoothstep` to the vector type of the other arguments, as GLSL's
`min(genType, float)`, `step(float, genType)` and similar overloads mean. It
used to pass the scalar as it was, which naga rejected (#4498).
