# CrossTL translation matrix: Slang to SPIR-V

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature SPIR-V has too straight to SPIR-V, and spirv-val checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 91.0 %: 443 supported, 12 refused, 32 wrong, 0 untested.

## Gaps

<details>
<summary>44 gaps: 12 refused, 32 wrong</summary>

Wrong, by cause:

- `on vulkan-lavapipe as SPIR-V from slangc, the translation binds other resources than the native probe` (25): `uniform-array`, `uniform-bool`, `uniform-bool-vector`, `uniform-buffer`, `uniform-cbuffer-vector`, `uniform-in-helper`, `uniform-integer`, `uniform-integer-vector`, `uniform-matrix`, `uniform-matrix2`, `uniform-matrix3`, `uniform-members`, `uniform-non-square-matrix`, `uniform-row-major-array`, `uniform-row-major-matrix`, `uniform-row-major-non-square`, `uniform-std140-array`, `uniform-std140-vector`, `uniform-struct`, `uniform-struct-array`, `uniform-struct-copy`, `uniform-unsigned`, `uniform-vector`, `uniform-vector-array`, `uniform-vector-packing`
- `on vulkan-lavapipe as SPIR-V from slangc, the translation computes different results from the native probe`: `QuadReadLaneAt`
- `the translation loses Decoration.Centroid`: `centroid-interpolation`
- `the translation loses Decoration.Flat`: `flat-interpolation`
- `the translation loses Decoration.NoPerspective`: `noperspective-interpolation`
- `the translation loses Decoration.Sample, Capability.SampleRateShading`: `sample-interpolation`
- `the translation loses ImageFormat.R32ui`: `integer-storage-image`
- `the translation loses ImageFormat.Rgba8`: `unorm-storage-image`

Refused, by diagnostic:

- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to '<feature>' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering` (4): `asdouble`, `max3`, `min3`, `rint`
- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @SV_PointCoord is not a Direct3D system value; the HLSL system values are SV_Barycentrics, SV_ClipDistance, SV_Coverage, SV_CullDistance, SV_CullPrimitive, SV_Depth, SV_DepthGreaterEqual, SV_DepthLessEqual, SV_DispatchGrid, SV_DispatchThreadID, SV_DomainLocation, SV_GSInstanceID, SV_GroupID, SV_Grou`: `point-coordinate`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`
- `project.translate.unsupported-feature: UnsupportedSPIRVFeatureError: SPIR-V requires a concrete fixed-array extent for local variable t; could not resolve 'N'`: `constant-array-size`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to '<expression>' to SPIR-V: Vulkan SPIR-V has no function pointers, so a call through a function table or other callee expression cannot be lowered`: `texel-buffer`
- `project.translate.vulkan-builtin-unsupported: SPIRVBuiltinLoweringError: Cannot lower call to 'IsHelperLane' to SPIR-V: it is neither a declared function nor a built-in with a SPIR-V lowering`: `helper-invocation`
- `project.translate.vulkan-lowering-unsupported: SpirvLoweringRefused: SPIR-V cannot lower N construct: Unknown type RWBuffer<float4>, using float as default (target vulkan) [project.translate.vulkan-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for vulkan (target vulkan) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

</details>
