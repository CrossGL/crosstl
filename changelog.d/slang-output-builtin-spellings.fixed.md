Slang output now reads the subgroup built-ins from Slang's wave intrinsics (`WaveGetLaneCount()`, `WaveGetLaneIndex()`,
the `WaveGetLane*Mask()` masks, and `WaveGetNumWaves()`/`WaveGetWaveIndex()` in compute-class stages), writes
`select` with the condition first (and the Slang importer writes it back in CrossGL order), declares
`gl_PointSize` as `SV_PointSize`, writes `layout(early_fragment_tests)` as `[earlydepthstencil]`, selects
`SV_DepthGreaterEqual`/`SV_DepthLessEqual` for conservative depth layouts, declares rasterizer-ordered buffers and
images with the `RasterizerOrdered*` types, and keeps an `hlsl_cbuffer` constant buffer's HLSL offsets with
`packoffset` (refusing a member HLSL packs inside std140 padding).
