HLSL output writes through a dynamic matrix column (`m[k].y = v`, `m[k].x += v`, `m[k].y++`, an `out` or
`inout` argument `m[k]` or `m[k].y`, the result argument of `modf` or `frexp`) into the matrix itself. It used to
write them through `transpose(m)[k]`, which DXC accepts but applies to a temporary, so the write was lost. A
dynamic column write whose value is used is refused.
