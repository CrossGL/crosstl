`mix(x, y, a)` whose selector is a call to `any`, `all`, `isnan` or `isinf` now
selects on Metal, HLSL, Slang, WGSL, CUDA and HIP instead of blending, so an
unselected infinite or NaN operand no longer turns the result into NaN, and
naga accepts the WGSL output. The generators gave these calls no result type,
so the boolean `mix` lowering did not see a `bool` or `bvecN` selector. CUDA
and HIP cover the scalar forms (#4538).
