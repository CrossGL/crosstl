# CrossTL translation matrix: Metal to Slang

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature Slang has too straight to Slang, and slangc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 92.5 %: 395 supported, 27 refused, 5 wrong, 0 untested.

## Gaps

<details>
<summary>32 gaps: 27 refused, 5 wrong</summary>

Wrong, by cause:

- `error[E31211]: derivative group linear thread count error: compute derivative group linear requires total thread dispatch count to be at a multiple of N`: `texture-1d-array`
- `error[E39999]: no overload for 'float4' applicable to arguments of type ()`: `zero-initializer`
- `Semantic must be defined for all outputs of an entry function or patch constant function`: `tessellation-patch-input`
- `the translation loses Capability.RuntimeDescriptorArray`: `unsized-array-of-textures`
- `the translation loses Capability.ShaderNonUniform`: `array-of-textures-nonuniform-index`

Refused, by diagnostic:

- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to atomicThreadFence: atomicThreadFence is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]` (5): `atomic_exchange`, `atomic_fetch_add`, `atomic_fetch_and`, `atomic_fetch_or`, `atomic_fetch_xor`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to <feature>: <feature> is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]` (4): `dfdx`, `dfdy`, `simd_is_first`, `simd_shuffle_rotate_down`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang has no system value for the built-in @<feature>; it would become a user varying, not the built-in (target slang) [project.translate.slang-lowering-unsupported]` (2): `num-subgroups`, `subgroup-id`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'CrossGLMetalVectorIndex_vec2_add_assign_resource_r_index.data' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); Slang structured buffer: buffer_load requires StructuredBuffer o` (2): `buffer-row-major-matrix`, `buffer-row-major-non-square`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang compute entry '<probe>' calls SampleBias but its 1x1x1 thread group is not a derivative group: a group with y == N needs x to be a multiple of four (linear quads); slangc rejects the derivatives with error E31211 or E31210 (target slang) [project.translate.slang-l`: `texture-1d`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang image access: imageStore requires an image resource (target slang) [project.translate.slang-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: Slang sampled texture: texelFetch requires a non-shadow texel-fetchable sampled texture resource (target slang) [project.translate.slang-lowering-unsupported]`: `texel-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to atomicCompareExchangeWeak: atomicCompareExchangeWeak is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `atomic_compare_exchange_weak_explicit`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: call to simd_is_helper_thread: simd_is_helper_thread is a built-in function of another language, and Slang declares no function of that name (target slang) [project.translate.slang-lowering-unsupported]`: `helper-invocation`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: pointer declaration: 'probe.p' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)") (target slang) [project.translate.slang-lowering-unsupported]`: `pointer-local`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N construct: resource parameter '<probe>' is coherent: slangc allows no coherent or volatile parameter (E31201), so its accesses would lose that behaviour (target slang) [project.translate.slang-lowering-unsupported]`: `coherent-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: call to atomicThreadFence: atomicThreadFence is a built-in function of another language, and Slang declares no function of that name; call to atomicCompareExchangeWeak: atomicCompareExchangeWeak is a built-in function of another language, and Slang declares no functi`: `atomic_compare_exchange_weak`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'CrossGLMetalVectorIndex_vec2_add_assign_resource_index.data' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); Slang structured buffer: buffer_load requires StructuredBuffer or`: `matrix-buffer`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'CrossGLMetalVectorIndex_vec2_add_assign_resource_r_index_index.data' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); Slang structured buffer: buffer_load requires StructuredBu`: `buffer-row-major-array`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'fetch.iin' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'put.outp' points into device memory, and a bound Slang buffer has no device address (Slang Use`: `buffer-in-helper`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'probe.p' points into device memory, and a bound Slang buffer has no device address (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'probe.q' points into device memory, and a bound Slang buffer has no device address (Slang User's`: `pointer-arithmetic`
- `project.translate.slang-lowering-unsupported: SlangLoweringRefused: Slang cannot lower N constructs: pointer declaration: 'stash.tile' points into threadgroup memory, and Slang's & of groupshared memory gives a device pointer (Slang User's Guide, "Pointers (limited)"); pointer declaration: 'exchange.tile' points into threadgroup memory, and Slang's & of groupshared`: `shared-memory-in-helper`
- `project.translate.slang-refused: SlangRefusal: Unsupported gl_BaseVertex stage parameter semantic for Slang codegen; expected int type`: `base-vertex`

</details>
