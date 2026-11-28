# CrossTL translation matrix: GLSL ES to Metal

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL ES probe whose feature Metal has too straight to Metal, and metal checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 100.0 %: 383 supported, 0 refused, 0 wrong, 6 untested.

## Gaps

<details>
<summary>6 untested: graphics 4, type 2</summary>

- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance`: `clip-distance`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
