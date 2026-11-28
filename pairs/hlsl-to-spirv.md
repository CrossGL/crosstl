# CrossTL translation matrix: HLSL to SPIR-V

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HLSL probe whose feature SPIR-V has too straight to SPIR-V, and spirv-val checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.6 %: 440 supported, 8 refused, 3 wrong, 0 untested.

## Gaps

<details>
<summary>11 gaps: 8 refused, 3 wrong</summary>

Wrong, by cause:

- `on vulkan-lavapipe as SPIR-V from dxc, the translation computes different results from the native probe`: `cross`
- `the translation calls no NMax and tests no operand for NaN`: `max`
- `the translation calls no NMin and tests no operand for NaN`: `min`

Refused, by diagnostic:

- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to '<feature>' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering` (4): `f16tof32`, `f32tof16`, `pack_clamp_s8`, `pack_clamp_u8`
- `project.translate.vulkan-lowering-unsupported: SpirvLoweringRefused: SPIR-V cannot lower N construct: atomicCompSwap currently supports only int or uint buffer members (target vulkan) [project.translate.vulkan-lowering-unsupported]` (2): `InterlockedCompareExchangeFloatBitwise`, `InterlockedCompareStoreFloatBitwise`
- `project.translate.unsupported-feature: UnsupportedSPIRVFeatureError: SPIR-V requires a concrete fixed-array extent for local variable t; could not resolve 'N'`: `constant-array-size`
- `project.translate.vulkan-lowering-unsupported: SpirvLoweringRefused: SPIR-V cannot lower N construct: atomicExchange currently supports only int or uint buffer members (target vulkan) [project.translate.vulkan-lowering-unsupported]`: `InterlockedExchange`

</details>
