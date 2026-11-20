WGSL output replicates a scalar argument of `min`, `max`, `step` and
`smoothstep` to the vector type of the other arguments, as GLSL's
`min(genType, float)`, `step(float, genType)` and similar overloads mean,
including a scalar returned by `length`, `distance`, `dot` or a component-wise
built-in such as `abs` or `sqrt`. It used to pass the scalar as it was, which
naga rejected (#4498).
