GLSL output keeps a compute, mesh or task stage's `layout(derivative_group_quadsKHR) in;`
or `layout(derivative_group_linearKHR) in;` (and the NV forms), with the extension
that declares it. It used to drop the layout, so quad operations in an 8x8 workgroup
read four consecutive subgroup invocations instead of 2x2 blocks.
