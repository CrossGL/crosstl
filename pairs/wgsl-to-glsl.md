# CrossTL translation matrix: WGSL to GLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature GLSL has too straight to GLSL, and glslangValidator checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.8 %: 353 supported, 5 refused, 3 wrong, 4 untested.

## Gaps

<details>
<summary>8 gaps: 5 refused, 3 wrong</summary>

Wrong, by cause:

- `'assign' : cannot convert from ' const N-element array of float' to ' out unsized N-element array of float ClipDistance'`: `clip-distance`
- `'assign' : l-value required "counter" (can't modify a uniform)`: `global-variable`
- `'assign' : l-value required "table" (can't modify a uniform)`: `global-array`

Refused, by diagnostic:

- `project.translate.opengl-workgroup-pointer-unsupported: OpenGLWorkgroupPointerError: OpenGL cannot reassign workgroup pointer '<probe>'` (5): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`, `shared-struct-array`, `shared-vector-array`

</details>

<details>
<summary>4 untested: function 4</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
