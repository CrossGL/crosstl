# CrossTL translation matrix: WGSL to HIP

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature HIP has too straight to HIP, and hipcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.2 %: 210 supported, 6 refused, 0 wrong, 3 untested.

## Gaps

<details>
<summary>6 gaps: 6 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.array-constructor-unsupported: ArrayConstructorUnsupportedError: array constructor 'float[N]' is assigned to an array; the target builds arrays only in declarations and aggregate initializers, so the value cannot be built where the source builds it (target hip) [project.translate.array-constructor-unsupported]` (3): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`
- `project.translate.array-constructor-unsupported: ArrayConstructorUnsupportedError: array constructor 'Pair[N]' is assigned to an array; the target builds arrays only in declarations and aggregate initializers, so the value cannot be built where the source builds it (target hip) [project.translate.array-constructor-unsupported]`: `shared-struct-array`
- `project.translate.array-constructor-unsupported: ArrayConstructorUnsupportedError: array constructor 'float[N][N]' is assigned to an array; the target builds arrays only in declarations and aggregate initializers, so the value cannot be built where the source builds it (target hip) [project.translate.array-constructor-unsupported]`: `shared-2d-array`
- `project.translate.array-constructor-unsupported: ArrayConstructorUnsupportedError: array constructor 'vec4[N]' is assigned to an array; the target builds arrays only in declarations and aggregate initializers, so the value cannot be built where the source builds it (target hip) [project.translate.array-constructor-unsupported]`: `shared-vector-array`

</details>

<details>
<summary>3 untested: function 3</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`

</details>
