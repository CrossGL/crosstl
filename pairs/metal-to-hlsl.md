# CrossTL translation matrix: Metal to HLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature HLSL has too straight to HLSL, and dxc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 95.6 %: 372 supported, 12 refused, 5 wrong, 0 untested.

## Gaps

<details>
<summary>17 gaps: 12 refused, 5 wrong</summary>

Wrong, by cause:

- `Semantic must be defined for all outputs of an entry function or patch constant function` (3): `tessellation-domain-quads`, `tessellation-evaluation`, `tessellation-patch-input`
- `object 'Texture2D' is not allowed in builtin template parameters`: `unsized-array-of-textures`
- `the translation loses Capability.ShaderNonUniform`: `array-of-textures-nonuniform-index`

Refused, by diagnostic:

- `project.translate.directx-atomic-fence-unsupported: DirectXAtomicFenceLoweringError: Cannot lower CrossGL atomicThreadFence to HLSL without changing its semantics (flags=mem_device | mem_threadgroup | mem_threadgroup_imageblock | mem_texture | mem_object_data, order=memory_order_seq_cst, scope=thread_scope_device): unsupported thread scope` (6): `atomic_compare_exchange_weak`, `atomic_exchange`, `atomic_fetch_add`, `atomic_fetch_and`, `atomic_fetch_or`, `atomic_fetch_xor`
- `project.translate.directx-lowering-unsupported: DirectXLoweringRefused: DirectX cannot lower N construct: function: call to '<feature>', a built-in function of another shading language that DirectX does not define and the program does not declare (target directx) [project.translate.directx-lowering-unsupported]` (5): `dfdx`, `dfdy`, `max3`, `min3`, `simd_is_first`
- `project.translate.directx-lowering-unsupported: DirectXLoweringRefused: DirectX cannot lower N construct: function: call to 'simd_is_helper_thread', a built-in function of another shading language that DirectX does not define and the program does not declare (target directx) [project.translate.directx-lowering-unsupported]`: `helper-invocation`

</details>
