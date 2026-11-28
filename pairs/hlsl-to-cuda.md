# CrossTL translation matrix: HLSL to CUDA

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HLSL probe whose feature CUDA has too straight to CUDA, and nvcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 98.8 %: 237 supported, 3 refused, 0 wrong, 0 untested.

## Gaps

<details>
<summary>3 gaps: 3 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: CUDA structured buffer atomic: atomicCompSwap on RWStructuredBuffer<float> requires supported scalar int/uint target (target cuda) [project.translate.cuda-lowering-unsupported]` (2): `InterlockedCompareExchangeFloatBitwise`, `InterlockedCompareStoreFloatBitwise`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: CUDA imageStore on storage image 'img': @texel(unorm, vec4) leaves the size of a texel to the bound view, which a surface does not have (target cuda) [project.translate.cuda-lowering-unsupported]`: `unorm-storage-image`

</details>
