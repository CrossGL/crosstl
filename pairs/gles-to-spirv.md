# CrossTL translation matrix: GLSL ES to SPIR-V

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL ES probe whose feature SPIR-V has too straight to SPIR-V, and spirv-val checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 99.4 %: 464 supported, 3 refused, 0 wrong, 7 untested.

## Gaps

<details>
<summary>3 gaps: 3 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'interpolate_at_centroid' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `interpolateAtCentroid`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'interpolate_at_offset' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `interpolateAtOffset`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'interpolate_at_sample' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `interpolateAtSample`

</details>

<details>
<summary>7 untested: graphics 5, type 2</summary>

- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
