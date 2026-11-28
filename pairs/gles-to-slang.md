# CrossTL translation matrix: GLSL ES to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL ES probe whose feature Slang has too straight to Slang, and slangc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 96.8 %: 423 supported, 11 refused, 3 wrong, 7 untested.

## Gaps

<details>
<summary>14 gaps: 11 refused, 3 wrong</summary>

Wrong, by cause:

- `error[E30015]: undefined identifier '<feature>'` (2): `barycentric-coordinate`, `shading-rate`
- `error[E30047]: argument must be l-value: argument passed to parameter 'N' must be l-value`: `buffer-element-argument`

Refused, by diagnostic:

- `project.translate.slang-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array 'position' is not a resource array: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of` (7): `tessellation-control`, `tessellation-domain-isolines`, `tessellation-domain-quads`, `tessellation-evaluation`, `tessellation-level-input`, `tessellation-patch-input`, `tessellation-patch-output`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang image access: imageStore requires an image resource (target slang) [project.translate.slang-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to interpolate_at_centroid: interpolate_at_centroid is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `interpolateAtCentroid`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to interpolate_at_sample: interpolate_at_sample is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `interpolateAtSample`

</details>

<details>
<summary>7 untested: graphics 5, type 2</summary>

- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `untested in the gles input row: the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
