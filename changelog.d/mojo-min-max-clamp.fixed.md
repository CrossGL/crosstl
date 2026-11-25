Mojo output writes `min`, `max` and `clamp` with Mojo's standard `min` and
`max` for every integer and float scalar and vector, converting each operand
to the result type, instead of declaring module-level overloads that hid the
standard ones for the types they did not list. `min(ivec2, ivec2)` and the
integer workgroup minimum and maximum now compile.
