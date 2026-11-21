SPIR-V output honours a CrossGL compute, mesh or task stage's `layout(derivative_group_quadsKHR) in;` or
`layout(derivative_group_linearKHR) in;` (the NV spellings too): it declares `DerivativeGroupQuadsKHR` or
`DerivativeGroupLinearKHR`, which form the quads of the stage's derivatives and quad operations, and refuses a
workgroup the group cannot divide. A declared group takes precedence over the group derived for derivatives, and a
stage with quad operations and no layout keeps subgroup quads (#4840). SPIR-V input imports the two execution modes
as these layouts. Before, a stage that declared 2x2 quads read subgroup quads in `QuadReadAcrossY` on lavapipe, and an
8x8 stage that declared the linear group and took derivatives declared the 2x2 group.
