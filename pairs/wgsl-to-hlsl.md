# CrossTL translation matrix: WGSL to HLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature HLSL has too straight to HLSL, and dxc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 99.1 %: 344 supported, 0 refused, 3 wrong, 4 untested.

## Gaps

<details>
<summary>3 gaps: 0 refused, 3 wrong</summary>

Wrong, by cause:

- `on vulkan-lavapipe as SPIR-V from naga and dxc, the translation computes different results from the native probe` (2): `determinant`, `uniform-std140-vector`
- `the translation loses Decoration.DescriptorSet`: `descriptor-set`

</details>

<details>
<summary>4 untested: function 4</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
