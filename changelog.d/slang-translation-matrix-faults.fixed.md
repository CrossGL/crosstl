Slang output: fix the faults that the translation matrix found in HLSL, Metal, GLSL and CUDA sources translated to
Slang. Switch cases that declare the same local name keep their own scopes; MSL's two-operand `sincos`, empty
initializers (`float4 v = {};`), `packDouble2x32`/`unpackDouble2x32`, entry-parameter interpolation modifiers,
`gl_BaryCoordEXT`, `gl_BaryCoordNoPerspEXT` and `gl_ShadingRateEXT` now compile with slangc; a GLSL buffer element
passed to an `inout` or `out` parameter is copied in and out; `IncrementCounter`/`DecrementCounter` use an explicit
counter buffer; compute-stage implicit-LOD samples read the base level; inferred declarations are `var` and C's
`char`/`short` are Slang's 8- and 16-bit integers. Slang output now refuses, instead of writing, a built-in type of
another language (`cudaTextureObject_t`), binary64 math that slangc writes as 32-bit GLSL.std.450 instructions, and
a resource member of an element of an array or buffer of structures.
