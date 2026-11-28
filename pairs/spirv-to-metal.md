# CrossTL translation matrix: SPIR-V to Metal

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native SPIR-V probe whose feature Metal has too straight to Metal, and metal checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 100.0 %: 411 supported, 0 refused, 0 wrong, 9 untested.

## Gaps

<details>
<summary>9 untested: control-flow 3, operator 1, type 5</summary>

- `dxc or spirv-dis is not installed` (7): `bool-bitwise`, `default-argument`, `enum-constant`, `generic-function`, `integer-condition`, `struct-method`, `type-alias`
- `slangc or spirv-dis is not installed` (2): `pointer-arithmetic`, `pointer-local`

</details>
