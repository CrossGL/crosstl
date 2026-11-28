# CrossTL translation matrix: WebGL to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WebGL probe whose feature WGSL has too straight to WGSL, and naga checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 98.8 %: 253 supported, 0 refused, 3 wrong, 0 untested.

## Gaps

<details>
<summary>3 gaps: 0 refused, 3 wrong</summary>

Wrong, by cause:

- `no definition in scope for identifier: '<feature>'` (3): `front-facing`, `instance-index`, `vertex-index`

</details>
