# CrossTL translation matrix: WGSL to Metal

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature Metal has too straight to Metal, and metal checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 100.0 %: 344 supported, 0 refused, 0 wrong, 4 untested.

## Gaps

<details>
<summary>4 untested: function 4</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
