Metal output reads the 2x2 quads of a compute, mesh or object thread group with
even x and y dimensions, as HLSL Shader Model 6.6 forms them (#4840). For a
width that is a power of two up to 16, `QuadReadAcrossY`,
`QuadReadAcrossDiagonal` and `QuadReadLaneAt` become SIMD-group shuffles to the
other row of the quad; other widths with even x and y are refused. It used to
use Metal's quad-groups of four consecutive threads, so `QuadReadAcrossY` in an
8x8 thread group read thread `gi ^ 2` instead of `gi ^ 8`. `QuadAny` and
`QuadAll` in compute-like stages now vote with the active threads of the quad
through `simd_ballot`, because Apple's compiler moves `quad_any` out of a branch
and lets the threads the branch excludes vote too.
