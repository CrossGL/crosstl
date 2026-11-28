# CrossTL translation matrix: Slang to CUDA

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature CUDA has too straight to CUDA, and nvcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.6 %: 247 supported, 6 refused, 0 wrong, 0 untested.

## Gaps

<details>
<summary>6 gaps: 6 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: function: call to '<feature>', a built-in function of another shading language that CUDA does not define and the program does not declare (target cuda) [project.translate.cuda-lowering-unsupported]` (2): `max3`, `min3`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: CUDA imageStore on storage image 'img': @texel(unorm, vec4) leaves the size of a texel to the bound view, which a surface does not have (target cuda) [project.translate.cuda-lowering-unsupported]`: `unorm-storage-image`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: type: the built-in type 'Buffer' has no CUDA spelling (target cuda) [project.translate.cuda-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareStoreFloatBitwise needs a float location (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareStoreFloatBitwise`
- `project.translate.slang-import-unsupported: SlangImportRefused: CrossGL cannot represent N Slang construct: Slang InterlockedCompareExchangeFloatBitwise takes N arguments, not N (target cgl) [project.translate.slang-import-unsupported]`: `InterlockedCompareExchangeFloatBitwise`

</details>
