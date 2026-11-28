# CrossTL translation matrix: SPIR-V to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native SPIR-V probe whose feature WGSL has too straight to WGSL, and naga checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 98.4 %: 358 supported, 3 refused, 3 wrong, 1 untested.

## Gaps

<details>
<summary>6 gaps: 3 refused, 3 wrong</summary>

Wrong, by cause:

- `no matching call to 'bitcast<u32>(vec2<i32>)'` (2): `vector-conversion`, `vector-shift`
- `no matching call to 'bitcast<i32>(vec3<f32>)'`: `OpBitcast`

Refused, by diagnostic:

- `project.translate.wgsl-numeric-type-unsupported: WGSLNumericTypeUnsupportedError: WGSL target cannot represent N-bit integer type 'int64_t': WGSL scalars are bool, i32, u32, f32 and f16 (WGSL section N.N), and narrowing the value would change its range and precision (target wgsl) [project.translate.wgsl-numeric-type-unsupported]` (2): `integer-literal-limits`, `unsigned-switch`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support mix with a N-component selector and N-component values`: `vector-conditional`

</details>

<details>
<summary>1 untested: function 1</summary>

- `the target's own probe is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `OpGroupNonUniformElect`

</details>
