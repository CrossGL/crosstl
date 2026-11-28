# CrossTL translation matrix: GLSL ES to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL ES probe whose feature WGSL has too straight to WGSL, and naga checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 98.6 %: 349 supported, 0 refused, 5 wrong, 2 untested.

## Gaps

<details>
<summary>5 gaps: 0 refused, 5 wrong</summary>

Wrong, by cause:

- `no definition in scope for identifier: '<feature>'` (5): `front-facing`, `instance-index`, `primitive-id`, `sample-index`, `vertex-index`

</details>

<details>
<summary>2 untested: function 1, graphics 1</summary>

- `the target's own probe is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance`: `clip-distance`

</details>
