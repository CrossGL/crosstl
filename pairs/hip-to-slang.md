# CrossTL translation matrix: HIP to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HIP probe whose feature Slang has too straight to Slang, and slangc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 96.8 %: 276 supported, 8 refused, 1 wrong, 0 untested.

## Gaps

<details>
<summary>9 gaps: 8 refused, 1 wrong</summary>

Wrong, by cause:

- `line N: GLSL.std.N Atan: expected Result Type to be a N or N-bit scalar or vector float type`: `atan`

Refused, by diagnostic:

- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_libm_sin on a N-bit floating-point operand has no helper for slang: the helper takes binary32 values only (target slang) [project.translate.rule-inexpressible]` (2): `sin`, `sincos`
- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_libm_cos on a N-bit floating-point operand has no helper for slang: the helper takes binary32 values only (target slang) [project.translate.rule-inexpressible]`: `cos`
- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_round_away on a N-bit floating-point operand has no helper for slang: the helper takes binary32 values only (target slang) [project.translate.rule-inexpressible]`: `round`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: pointer declaration: 'probe.p' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)") (target slang) [project.translate.slang-lowering-unsupported]`: `pointer-local`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'fetch.iin' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'put.outp' points into device memory, and a bound Slang buffer has no device address (Slang Use`: `buffer-in-helper`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'probe.p' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'probe.q' points into device memory, and a bound Slang buffer has no device address (Slang User's`: `pointer-arithmetic`

</details>
