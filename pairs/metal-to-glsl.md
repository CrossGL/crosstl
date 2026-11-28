# CrossTL translation matrix: Metal to GLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature GLSL has too straight to GLSL, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.4 %: 417 supported, 8 refused, 3 wrong, 0 untested.

## Gaps

<details>
<summary>11 gaps: 8 refused, 3 wrong</summary>

Wrong, by cause:

- `'slots' : member of block cannot be or contain a sampler, image, or atomic_uint type`: `unsized-array-of-textures`
- `the translation loses Capability.ShaderNonUniform`: `array-of-textures-nonuniform-index`
- `the translation loses OpTypeSampler`: `shared-sampler`

Refused, by diagnostic:

- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: call to '<feature>', which is neither a GLSL built-in function nor a function the program declares (target opengl) [project.translate.opengl-lowering-unsupported]` (2): `dfdx`, `dfdy`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: wave intrinsic: <feature> has no OpenGL subgroup equivalent (target opengl) [project.translate.opengl-lowering-unsupported]` (2): `simd_is_first`, `simd_shuffle_rotate_down`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: tessellation patch parameter 'patch_points' of type 'OutputPatch<ControlPoint, 3>' has no GLSL stage variable form: GLSL reads control points through <feature>[] and per-vertex input arrays; GLSL tessellation_evaluation stage cannot declare HLSL control-point patch para` (2): `tessellation-evaluation`, `tessellation-patch-input`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: tessellation patch parameter 'patch_points' of type 'OutputPatch<ControlPoint, 4>' has no GLSL stage variable form: GLSL reads control points through gl_in[] and per-vertex input arrays; GLSL tessellation_evaluation stage cannot declare HLSL control-point patch para`: `tessellation-domain-quads`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from '<unresolved>' to 'bool': boolean context source type unresolved`: `helper-invocation`

</details>
