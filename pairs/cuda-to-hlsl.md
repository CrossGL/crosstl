# CrossTL translation matrix: CUDA to HLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native CUDA probe whose feature HLSL has too straight to HLSL, and dxc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 98.9 %: 262 supported, 3 refused, 0 wrong, 0 untested.

## Gaps

<details>
<summary>3 gaps: 3 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_libm_sin on a N-bit floating-point operand has no helper for directx: the helper takes binary32 values only (target directx) [project.translate.rule-inexpressible]` (2): `sin`, `sincos`
- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_libm_cos on a N-bit floating-point operand has no helper for directx: the helper takes binary32 values only (target directx) [project.translate.rule-inexpressible]`: `cos`

</details>
