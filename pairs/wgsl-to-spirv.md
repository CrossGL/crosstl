# CrossTL translation matrix: WGSL to SPIR-V

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature SPIR-V has too straight to SPIR-V, and spirv-val checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 99.4 %: 360 supported, 2 refused, 0 wrong, 4 untested.

## Gaps

<details>
<summary>2 gaps: 2 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to '<feature>' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering` (2): `pack4xI8Clamp`, `pack4xU8Clamp`

</details>

<details>
<summary>4 untested: function 4</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
