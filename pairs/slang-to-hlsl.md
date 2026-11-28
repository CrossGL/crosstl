# CrossTL translation matrix: Slang to HLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature HLSL has too straight to HLSL, and dxc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 92.7 %: 493 supported, 11 refused, 28 wrong, 0 untested.

## Gaps

<details>
<summary>39 gaps: 11 refused, 28 wrong</summary>

Wrong, by cause:

- `on vulkan-lavapipe as SPIR-V from slangc and dxc, the translation binds other resources than the native probe` (26): `read-only-buffer`, `uniform-array`, `uniform-bool`, `uniform-bool-vector`, `uniform-buffer`, `uniform-cbuffer-vector`, `uniform-in-helper`, `uniform-integer`, `uniform-integer-vector`, `uniform-matrix`, `uniform-matrix2`, `uniform-matrix3`, `uniform-members`, `uniform-non-square-matrix`, `uniform-row-major-array`, `uniform-row-major-matrix`, `uniform-row-major-non-square`, `uniform-std140-array`, `uniform-std140-vector`, `uniform-struct`, `uniform-struct-array`, `uniform-struct-copy`, `uniform-unsigned`, `uniform-vector`, `uniform-vector-array`, `uniform-vector-packing`
- `validation errors`: `GetAttributeAtVertex`
- `use of undeclared identifier 'MaybeReorderThread'`: `MaybeReorderThread`

Refused, by diagnostic:

- `project.translate.directx-lowering-unsupported: DirectXLoweringRefused: DirectX cannot lower N construct: function: call to '<feature>', a built-in function of another shading language that DirectX does not define and the program does not declare (target directx) [project.translate.directx-lowering-unsupported]` (3): `divide`, `max3`, `min3`
- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target directx) [project.translate.undeclared-function]` (3): `ReorderThread`, `ReportHitOptix`, `f32tof16_`
- `project.translate.directx-contextual-conversion-unrepresentable: DirectXContextualConversionError: DirectX NonUniformResourceIndex index argument must be scalar int or uint, got float`: `NonUniformResourceIndex`
- `project.translate.directx-refused: DirectXRefusal: DirectX ray_intersection ReportHit hit attribute argument must use a user-defined struct type or BuiltInTriangleIntersectionAttributes, got int`: `ReportHit`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for directx (target directx) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

</details>
