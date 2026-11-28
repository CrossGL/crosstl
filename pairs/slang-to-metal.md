# CrossTL translation matrix: Slang to Metal

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature Metal has too straight to Metal, and metal checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 98.6 %: 426 supported, 5 refused, 1 wrong, 0 untested.

## Gaps

<details>
<summary>6 gaps: 5 refused, 1 wrong</summary>

Wrong, by cause:

- `no member named 'Load' in 'metal::texture_buffer<float, metal::access::read, metal::memory_coherence_threadgroup>'`: `texel-buffer`

Refused, by diagnostic:

- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @SV_PointCoord is not a Direct3D system value; the HLSL system values are SV_Barycentrics, SV_ClipDistance, SV_Coverage, SV_CullDistance, SV_CullPrimitive, SV_Depth, SV_DepthGreaterEqual, SV_DepthLessEqual, SV_DispatchGrid, SV_DispatchThreadID, SV_DomainLocation, SV_GSInstanceID, SV_GroupID, SV_Grou`: `point-coordinate`
- `project.translate.metal-lowering-unsupported: MetalLoweringRefused: Metal cannot lower N construct: function: call to 'IsHelperLane', a built-in function of another shading language that Metal does not define and the program does not declare (target metal) [project.translate.metal-lowering-unsupported]`: `helper-invocation`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for metal (target metal) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

</details>
