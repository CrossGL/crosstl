# CrossTL translation matrix: HLSL to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HLSL probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 95.2 %: 380 supported, 7 refused, 12 wrong, 7 untested.

## Gaps

<details>
<summary>19 gaps: 7 refused, 12 wrong</summary>

Wrong, by cause:

- `'image variables not declared 'writeonly' and without a format layout qualifier' : not supported with this profile: es` (6): `RWBuffer.Load`, `RWTexture2D.Load`, `RWTexture2DArray.Load`, `RWTexture3D.Load`, `array-of-images`, `read-write-image`
- `'image1D' : Reserved word.` (2): `RWTexture1D.GetDimensions`, `RWTexture1D.Load`
- `'image1DArray' : Reserved word.` (2): `RWTexture1DArray.GetDimensions`, `RWTexture1DArray.Load`
- `'+' : wrong operand types: no operation '+' exists that takes a left-hand operand of type ' temp highp float' and a right operand of type ' const int' (or there is no acceptable conversion)`: `implicit-conversion`
- `'assign' : cannot convert from ' temp int' to ' temp highp uint'`: `runtime-array-length`

Refused, by diagnostic:

- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: GLSL version requirement: type 'double' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec2' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec3' does not exist in GLSL ES; the t` (3): `WaveMatch`, `WaveMultiPrefixProduct`, `WaveMultiPrefixSum`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: built-in variable 'gl_BaseVertex' does not exist in GLSL ES; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `vertex-index`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: the 'noperspective' qualifier requires GL_NV_shader_noperspective_interpolation; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `noperspective-interpolation`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: GLSL version requirement: type 'dvec2' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'double' does not exist in GLSL ES; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `fma`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'float16' to 'float16': half unsupported profile`: `half`

</details>

<details>
<summary>7 untested: graphics 5, type 2</summary>

- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
