# CrossTL translation matrix: Slang to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 93.9 %: 428 supported, 21 refused, 7 wrong, 7 untested.

## Gaps

<details>
<summary>28 gaps: 21 refused, 7 wrong</summary>

Wrong, by cause:

- `'subgroupShuffle' : no matching overloaded function found` (3): `WaveBroadcastLaneAt`, `WaveReadLaneAt`, `WaveShuffle`
- `'image variables not declared 'writeonly' and without a format layout qualifier' : not supported with this profile: es` (2): `array-of-images`, `read-write-image`
- `'+' : wrong operand types: no operation '+' exists that takes a left-hand operand of type ' temp highp float' and a right operand of type ' const int' (or there is no acceptable conversion)`: `implicit-conversion`
- `'assign' : cannot convert from ' temp int' to ' temp highp uint'`: `runtime-array-length`

Refused, by diagnostic:

- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: GLSL version requirement: type 'double' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec2' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec3' does not exist in GLSL ES; the t` (5): `WaveMatch`, `WaveMultiPrefixExclusiveProduct`, `WaveMultiPrefixExclusiveSum`, `WaveMultiPrefixProduct`, `WaveMultiPrefixSum`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'f16vec2' to 'f16vec2': half unsupported profile` (3): `unpackHalf2x16ToHalf`, `unpackSnorm2x16ToHalf`, `unpackUnorm2x16ToHalf`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'f16vec4' to 'f16vec4': half unsupported profile` (2): `unpackSnorm4x8ToHalf`, `unpackUnorm4x8ToHalf`
- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @SV_PointCoord is not a Direct3D system value; the HLSL system values are SV_Barycentrics, SV_ClipDistance, SV_Coverage, SV_CullDistance, SV_CullPrimitive, SV_Depth, SV_DepthGreaterEqual, SV_DepthLessEqual, SV_DispatchGrid, SV_DispatchThreadID, SV_DomainLocation, SV_GSInstanceID, SV_GroupID, SV_Grou`: `point-coordinate`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: built-in variable 'gl_BaseVertex' does not exist in GLSL ES; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `vertex-index`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: the 'noperspective' qualifier requires GL_NV_shader_noperspective_interpolation; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `noperspective-interpolation`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: call through 'tex.Load': GLSL calls only named functions, constructors and length() (target opengl) [project.translate.opengl-lowering-unsupported]`: `texel-buffer`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: call to 'dst', which is neither a GLSL built-in function nor a function the program declares (target opengl) [project.translate.opengl-lowering-unsupported]`: `dst`
- `project.translate.opengl-metal-math-unrepresentable: OpenGLMetalMathError: OpenGL cannot preserve 'fabs' for operand types ('vec2',): unsupported-profile`: `fabs`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'half' to 'half': half unsupported profile`: `half`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from '<unresolved>' to 'bool': boolean context source type unresolved`: `helper-invocation`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for opengl (target opengl) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

</details>

<details>
<summary>7 untested: graphics 5, type 2</summary>

- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
