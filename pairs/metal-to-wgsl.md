# CrossTL translation matrix: Metal to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature WGSL has too straight to WGSL, and naga checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 88.2 %: 315 supported, 28 refused, 14 wrong, 1 untested.

## Gaps

<details>
<summary>42 gaps: 28 refused, 14 wrong</summary>

Wrong, by cause:

- `Entry point compute_main at Compute is invalid` (7): `sampled-texture`, `shared-sampler`, `texture-1d`, `texture-3d`, `texture-argument`, `texture-cube`, `texture-in-helper`
- `Entry point vertex_main at Vertex is invalid` (3): `instance-index`, `vertex`, `vertex-index`
- `[ERROR naga::front::wgsl::lower] Type Scalar(Scalar { kind: Sint, width: N }) passed to atomic op`: `atomic_compare_exchange_weak_explicit`
- `the type of 'a' is expected to be 'bool', but got 'i32'`: `bool-bitwise`
- `type can't be inferred`: `zero-initializer`
- `value nan cannot be represented as 'f32'`: `pow`

Refused, by diagnostic:

- `project.translate.wgsl-foreign-builtin-call: WGSLForeignBuiltinCallError: WGSL target cannot lower the call to atomicThreadFence(): atomicThreadFence is a built-in function of another language, and WGSL declares no built-in function of that name (target wgsl) [project.translate.wgsl-foreign-builtin-call]` (8): `atomic_compare_exchange_weak`, `atomic_exchange`, `atomic_fetch_add`, `atomic_fetch_and`, `atomic_fetch_or`, `atomic_fetch_xor`, `atomic_load`, `atomic_store`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target only supports pointer types for buffer/storage resources and helper parameters` (6): `buffer-in-helper`, `buffer-row-major-array`, `buffer-row-major-matrix`, `buffer-row-major-non-square`, `matrix-buffer`, `shared-memory-in-helper`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image2D yet; split texture/sampler/storage bindings are required` (3): `read-write-image`, `storage-image`, `unorm-storage-image`
- `project.translate.wgsl-foreign-builtin-call: WGSLForeignBuiltinCallError: WGSL target cannot lower the call to <feature>(): <feature> is a built-in function of another language, and WGSL declares no built-in function of that name (target wgsl) [project.translate.wgsl-foreign-builtin-call]` (2): `fabs`, `rint`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target splits the layer out of the textureLod() coordinate of an array texture, which reads the coordinate twice; assign the coordinate to a local variable first` (2): `texture-2d-array`, `texture-cube-array`
- `project.translate.wgsl-numeric-type-unsupported: WGSLNumericTypeUnsupportedError: WGSL target cannot represent N-bit integer type 'uint64': WGSL scalars are bool, i32, u32, f32 and f16 (WGSL section N.N), and narrowing the value would change its range and precision (target wgsl) [project.translate.wgsl-numeric-type-unsupported]`: `simd_ballot`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot pass a part of '<probe>' to inout parameter x of grow: the argument is not a whole local variable, and a ptr<function, T> argument must be a whole function-scope variable (WGSL N.N.N); copy it into a local variable around the call`: `buffer-element-argument`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot take the address of '<probe>' for pointer local 'p': WGSL takes the address only of a variable, and '<probe>' is a let value, a parameter, a constant, a uniform or a resource handle`: `pointer-local`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image1D yet; split texture/sampler/storage bindings are required`: `storage-image-1d`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image2DArray yet; split texture/sampler/storage bindings are required`: `storage-image-array`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image3D yet; split texture/sampler/storage bindings are required`: `storage-image-3d`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type uimage2D yet; split texture/sampler/storage bindings are required`: `integer-storage-image`

</details>

<details>
<summary>1 untested: function 1</summary>

- `the target's own probe is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `simd_is_first`

</details>
