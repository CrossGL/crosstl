# CrossTL translation matrix: GLSL to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL probe whose feature Slang has too straight to Slang, and slangc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 95.9 %: 470 supported, 17 refused, 3 wrong, 0 untested.

## Gaps

<details>
<summary>20 gaps: 17 refused, 3 wrong</summary>

Wrong, by cause:

- `error[E30015]: undefined identifier '<feature>'` (2): `barycentric-coordinate`, `shading-rate`
- `error[E30047]: argument must be l-value: argument passed to parameter 'N' must be l-value`: `buffer-element-argument`

Refused, by diagnostic:

- `project.translate.slang-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array 'position' is not a resource array: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of` (7): `tessellation-control`, `tessellation-domain-isolines`, `tessellation-domain-quads`, `tessellation-evaluation`, `tessellation-level-input`, `tessellation-patch-input`, `tessellation-patch-output`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: call to beginInvocationInterlockARB: beginInvocationInterlockARB is a built-in function of another language, and Slang declares no function of that name; call to endInvocationInterlockARB: endInvocationInterlockARB is a built-in function of another language, and Slan` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang image access: imageStore requires an image resource (target slang) [project.translate.slang-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to packDouble2x32: packDouble2x32 is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `packDouble2x32`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to fwidth_coarse: fwidth_coarse is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `fwidthCoarse`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to fwidth_fine: fwidth_fine is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `fwidthFine`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to interpolate_at_centroid: interpolate_at_centroid is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `interpolateAtCentroid`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to interpolate_at_sample: interpolate_at_sample is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `interpolateAtSample`
- `project.translate.slang-refused: SlangRefusal: Slang geometry stage must declare at most one stream output parameter`: `geometry-streams`

</details>
