# CrossTL translation matrix: HLSL to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HLSL probe whose feature Slang has too straight to Slang, and slangc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 96.7 %: 503 supported, 8 refused, 9 wrong, 0 untested.

## Gaps

<details>
<summary>17 gaps: 8 refused, 9 wrong</summary>

Wrong, by cause:

- `on vulkan-lavapipe as SPIR-V from dxc and slangc, the translation is rejected by the runtime` (2): `QuadAll`, `QuadAny`
- `error[E30200]: conflicting declaration: declaration of 't' conflicts with existing declaration`: `switch-case-scope`
- `error[E36107]: unavailable features in entry point: entrypoint '<probe>' uses features that are not available in 'compute' stage for 'spirv' compilation target`: `RWStructuredBuffer.IncrementCounter`
- `attribute patchconstantfunc only valid for HS.`: `tessellation-patch-input`
- `the translation loses Decoration.Centroid`: `centroid-interpolation`
- `the translation loses Decoration.Flat`: `flat-interpolation`
- `the translation loses Decoration.NoPerspective`: `noperspective-interpolation`
- `the translation loses Decoration.Sample, Capability.SampleRateShading`: `sample-interpolation`

Refused, by diagnostic:

- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang image access: imageStore requires an image resource (target slang) [project.translate.slang-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang structured buffer: atomicExchange requires scalar int or uint RWStructuredBuffer element (target slang) [project.translate.slang-lowering-unsupported]`: `InterlockedExchange`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang wave intrinsic: WaveActiveAllEqual returns bool but target expects bool2 (target slang) [project.translate.slang-lowering-unsupported]`: `WaveActiveAllEqual`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to interpolateAtOffset: interpolateAtOffset is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `EvaluateAttributeSnapped`
- `project.translate.slang-refused: SlangRefusal: Slang geometry stage must declare at most one stream output parameter`: `geometry-streams`
- `project.translate.slang-shader-execution-reordering-unsupported: ShaderExecutionReorderingUnsupportedError: dx::MaybeReorderThread (HLSL Shader Model N.N shader execution reordering) cannot be carried to slang: the Slang generator does not lower it to Slang's ReorderThread (target slang) [project.translate.slang-shader-execution-reordering-unsupported]`: `dx::MaybeReorderThread`
- `project.translate.slang-texture-residency-unsupported: TextureResidencyUnsupportedError: texture operation 'textureResident' reports tile residency, which slang cannot carry here: the slang emitter has no lowering for texture residency feedback (target slang) [project.translate.slang-texture-residency-unsupported]`: `CheckAccessFullyMapped`

</details>
