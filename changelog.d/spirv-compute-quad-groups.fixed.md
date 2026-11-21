SPIR-V output forms the quads of a compute, mesh or task entry's quad operations as HLSL Shader Model 6.6 does. An
entry that reaches `QuadReadAcrossX`, `QuadReadAcrossY`, `QuadReadAcrossDiagonal`, `QuadReadLaneAt`, `QuadAll` or
`QuadAny` declares `DerivativeGroupQuadsKHR` when its x and y dimensions are even and `DerivativeGroupLinearKHR` when
its invocation count is a multiple of four, as DXC does; an 8x8 `QuadReadAcrossY` read lane `gi ^ 2` on lavapipe
instead of `gi ^ 8`. `QuadAll` and `QuadAny` lower to SPV_KHR_quad_control's `OpGroupNonUniformQuadAllKHR` and
`OpGroupNonUniformQuadAnyKHR`.
