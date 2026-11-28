# CrossTL translation matrix: Metal to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 94.7 %: 378 supported, 18 refused, 3 wrong, 6 untested.

## Gaps

<details>
<summary>21 gaps: 18 refused, 3 wrong</summary>

Wrong, by cause:

- `'rgba32f' : format requires readonly or writeonly memory qualifier` (2): `array-of-images`, `read-write-image`
- `'+' : wrong operand types: no operation '+' exists that takes a left-hand operand of type ' temp highp float' and a right operand of type ' const int' (or there is no acceptable conversion)`: `implicit-conversion`

Refused, by diagnostic:

- `project.translate.opengl-refused: OpenGLRefusal: OpenGL target profile '#version N es' does not support GL_ARB_gpu_shader_int64; select a desktop GLSL profile for N-bit integer values` (5): `atomic_max`, `atomic_max_explicit`, `atomic_min`, `atomic_min_explicit`, `simd_ballot`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: call to '<feature>', which is neither a GLSL built-in function nor a function the program declares (target opengl) [project.translate.opengl-lowering-unsupported]` (2): `dfdx`, `dfdy`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: wave intrinsic: <feature> has no OpenGL subgroup equivalent (target opengl) [project.translate.opengl-lowering-unsupported]` (2): `simd_is_first`, `simd_shuffle_rotate_down`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: tessellation patch parameter 'patch_points' of type 'OutputPatch<ControlPoint, 3>' has no GLSL stage variable form: GLSL reads control points through <feature>[] and per-vertex input arrays; GLSL tessellation_evaluation stage cannot declare HLSL control-point patch para` (2): `tessellation-evaluation`, `tessellation-patch-input`
- `project.translate.opengl-fmod-unrepresentable: OpenGLFloatingRemainderError: OpenGL cannot preserve floating remainder for ('float', 'float'): unsupported-profile`: `float-remainder`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: the 'noperspective' qualifier requires GL_NV_shader_noperspective_interpolation; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `noperspective-interpolation`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: tessellation patch parameter 'patch_points' of type 'OutputPatch<ControlPoint, 4>' has no GLSL stage variable form: GLSL reads control points through gl_in[] and per-vertex input arrays; GLSL tessellation_evaluation stage cannot declare HLSL control-point patch para`: `tessellation-domain-quads`
- `project.translate.opengl-metal-math-unrepresentable: OpenGLMetalMathError: OpenGL cannot preserve 'select' for operand types ('vec3', 'vec3', 'bvec3'): unsupported-profile`: `select`
- `project.translate.opengl-metal-math-unrepresentable: OpenGLMetalMathError: OpenGL cannot preserve 'fabs' for operand types ('vec3',): unsupported-profile`: `fabs`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from '<unresolved>' to 'bool': boolean context source type unresolved`: `helper-invocation`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'float16' to 'float16': half unsupported profile`: `half`

</details>

<details>
<summary>6 untested: graphics 4, type 2</summary>

- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance`: `clip-distance`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
