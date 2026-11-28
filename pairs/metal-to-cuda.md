# CrossTL translation matrix: Metal to CUDA

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature CUDA has too straight to CUDA, and nvcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 94.6 %: 261 supported, 15 refused, 0 wrong, 0 untested.

## Gaps

<details>
<summary>15 gaps: 15 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: function: call to 'atomicThreadFence', a built-in function of another shading language that CUDA does not define and the program does not declare (target cuda) [project.translate.cuda-lowering-unsupported]` (7): `atomic_exchange`, `atomic_fetch_add`, `atomic_fetch_and`, `atomic_fetch_max`, `atomic_fetch_min`, `atomic_fetch_or`, `atomic_fetch_xor`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: function: call to '<feature>', a built-in function of another shading language that CUDA does not define and the program does not declare (target cuda) [project.translate.cuda-lowering-unsupported]` (2): `max3`, `min3`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: CUDA structured buffer atomic: atomicMax on RWStructuredBuffer<uint64> requires supported scalar int/uint target (target cuda) [project.translate.cuda-lowering-unsupported]`: `atomic_max_explicit`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: CUDA structured buffer atomic: atomicMin on RWStructuredBuffer<uint64> requires supported scalar int/uint target (target cuda) [project.translate.cuda-lowering-unsupported]`: `atomic_min_explicit`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: function: call to 'atomicCompareExchangeWeak', a built-in function of another shading language that CUDA does not define and the program does not declare (target cuda) [project.translate.cuda-lowering-unsupported]`: `atomic_compare_exchange_weak_explicit`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N constructs: CUDA structured buffer atomic: atomicMax on RWStructuredBuffer<uint64> requires supported scalar int/uint target; function: call to 'atomicThreadFence', a built-in function of another shading language that CUDA does not define and the program does not declare (target`: `atomic_max`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N constructs: CUDA structured buffer atomic: atomicMin on RWStructuredBuffer<uint64> requires supported scalar int/uint target; function: call to 'atomicThreadFence', a built-in function of another shading language that CUDA does not define and the program does not declare (target`: `atomic_min`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N constructs: function: call to 'atomicCompareExchangeWeak', a built-in function of another shading language that CUDA does not define and the program does not declare; function: call to 'atomicThreadFence', a built-in function of another shading language that CUDA does not define`: `atomic_compare_exchange_weak`

</details>
