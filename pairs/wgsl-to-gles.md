# CrossTL translation matrix: WGSL to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.4 %: 342 supported, 7 refused, 2 wrong, 5 untested.

## Gaps

<details>
<summary>9 gaps: 7 refused, 2 wrong</summary>

Wrong, by cause:

- `'uniform' : cannot initialize this type of qualifier` (2): `global-array`, `global-variable`

Refused, by diagnostic:

- `project.translate.opengl-workgroup-pointer-unsupported: OpenGLWorkgroupPointerError: OpenGL cannot reassign workgroup pointer '<probe>'` (5): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`, `shared-struct-array`, `shared-vector-array`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: the 'noperspective' qualifier requires GL_NV_shader_noperspective_interpolation; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `noperspective-interpolation`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'half' to 'half': half unsupported profile`: `half`

</details>

<details>
<summary>5 untested: function 4, graphics 1</summary>

- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance`: `clip-distance`
- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
