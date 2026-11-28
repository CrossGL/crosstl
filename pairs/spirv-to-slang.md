# CrossTL translation matrix: SPIR-V to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native SPIR-V probe whose feature Slang has too straight to Slang, and slangc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 90.5 %: 427 supported, 35 refused, 10 wrong, 0 untested.

## Gaps

<details>
<summary>45 gaps: 35 refused, 10 wrong</summary>

Wrong, by cause:

- `on vulkan-lavapipe as SPIR-V from slangc, the translation binds other resources than the native probe` (5): `num-subgroups`, `specialization-workgroup-size`, `subgroup-id`, `subgroup-invocation-id`, `subgroup-size`
- `on vulkan-lavapipe as SPIR-V from slangc, the translation computes different results from the native probe` (3): `buffer-row-major-array`, `buffer-row-major-matrix`, `buffer-row-major-non-square`
- `on vulkan-lavapipe as SPIR-V from slangc, the translation is rejected by the runtime` (2): `OpGroupNonUniformQuadAllKHR`, `OpGroupNonUniformQuadAnyKHR`

Refused, by diagnostic:

- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: textureLod returns float4 but target expects float (target slang) [project.translate.slang-lowering-unsupported]` (14): `array-of-textures`, `array-of-textures-loop-index`, `array-of-textures-nonuniform-index`, `sampled-texture`, `shared-sampler`, `texture-1d`, `texture-1d-array`, `texture-2d-array`, `texture-3d`, `texture-argument`, `texture-cube`, `texture-cube-array`, `texture-in-helper`, `unsized-array-of-textures`
- `project.translate.slang-refused: SlangRefusal: Slang tessellation factor built-ins <feature> and <feature> require an explicit patch constant function return value using SV_TessFactor and SV_InsideTessFactor` (7): `tessellation-control`, `tessellation-control-point-mode`, `tessellation-control-spacing-fractional-even`, `tessellation-control-spacing-fractional-odd`, `tessellation-control-winding-ccw`, `tessellation-level-input`, `tessellation-patch-output`
- `project.translate.slang-refused: SlangRefusal: Slang tessellation_evaluation stage requires a domain attribute` (4): `tessellation-domain-isolines`, `tessellation-domain-quads`, `tessellation-evaluation`, `tessellation-patch-input`
- `project.translate.slang-refused: SlangRefusal: Unsupported <feature> stage parameter semantic for Slang codegen; expected uint type` (4): `base-instance`, `layer`, `sample-mask-input`, `viewport-index`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: call to beginInvocationInterlockARB: beginInvocationInterlockARB is a built-in function of another language, and Slang declares no function of that name; call to endInvocationInterlockARB: endInvocationInterlockARB is a built-in function of another language, and Slan` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang image access: imageStore requires an image resource (target slang) [project.translate.slang-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to packDouble2x32: packDouble2x32 is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `PackDouble2x32`
- `project.translate.slang-refused: SlangRefusal: Slang geometry stage must declare at most one stream output parameter`: `geometry-streams`

</details>
