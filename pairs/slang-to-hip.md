# CrossTL translation matrix: Slang to HIP

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature HIP has too straight to HIP, and hipcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.3 %: 248 supported, 6 refused, 1 wrong, 0 untested.

## Gaps

<details>
<summary>7 gaps: 6 refused, 1 wrong</summary>

Wrong, by cause:

- `no template named 'Buffer'`: `texel-buffer`

Refused, by diagnostic:

- `project.translate.hip-lowering-unsupported: HipLoweringRefused: HIP cannot lower N construct: function: call to '<feature>', a built-in function of another shading language that HIP does not define and the program does not declare (target hip) [project.translate.hip-lowering-unsupported]` (2): `max3`, `min3`
- `project.translate.hip-lowering-unsupported: HipLoweringRefused: HIP cannot lower N construct: HIP imageStore on storage image 'img': @texel(unorm, vec4) leaves the size of a texel to the bound view, which a surface does not have (target hip) [project.translate.hip-lowering-unsupported]`: `unorm-storage-image`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for hip (target hip) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

</details>
