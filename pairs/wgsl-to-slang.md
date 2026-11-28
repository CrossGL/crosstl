# CrossTL translation matrix: WGSL to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature Slang has too straight to Slang, and slangc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 96.4 %: 346 supported, 2 refused, 11 wrong, 2 untested.

## Gaps

<details>
<summary>13 gaps: 2 refused, 11 wrong</summary>

Wrong, by cause:

- `error[E30011]: left of '=' is not an l-value` (2): `global-array`, `global-variable`
- `error[E30019]: type mismatch in expression: expected an expression of type 'float[N]', got 'float[N]'` (2): `array-of-arrays`, `shared-2d-array`
- `on vulkan-lavapipe as SPIR-V from naga and slangc, the translation binds other resources than the native probe`: `specialization-workgroup-size`
- `on vulkan-lavapipe as SPIR-V from naga and slangc, the translation computes different results from the native probe`: `determinant`
- `the translation loses Decoration.Centroid`: `centroid-interpolation`
- `the translation loses Decoration.DescriptorSet`: `descriptor-set`
- `the translation loses Decoration.Flat`: `flat-interpolation`
- `the translation loses Decoration.NoPerspective`: `noperspective-interpolation`
- `the translation loses Decoration.Sample, Capability.SampleRateShading`: `sample-interpolation`

Refused, by diagnostic:

- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang has no system value for the built-in @<feature>; it would become a user varying, not the built-in (target slang) [project.translate.slang-lowering-unsupported]` (2): `num-subgroups`, `subgroup-id`

</details>

<details>
<summary>2 untested: function 2</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
