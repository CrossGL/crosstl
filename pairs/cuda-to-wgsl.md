# CrossTL translation matrix: CUDA to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native CUDA probe whose feature WGSL has too straight to WGSL, and naga checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 91.0 %: 201 supported, 9 refused, 11 wrong, 0 untested.

## Gaps

<details>
<summary>20 gaps: 9 refused, 11 wrong</summary>

Wrong, by cause:

- `Entry point compute_main at Compute is invalid` (8): `sampled-texture`, `texture-1d`, `texture-2d-array`, `texture-3d`, `texture-argument`, `texture-cube`, `texture-cube-array`, `texture-in-helper`
- `value nan cannot be represented as 'f32'` (2): `cosf`, `sinf`
- `type can't be inferred`: `zero-initializer`

Refused, by diagnostic:

- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image2D yet; split texture/sampler/storage bindings are required` (2): `read-write-image`, `storage-image`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type uimage2D yet; split texture/sampler/storage bindings are required` (2): `integer-storage-image`, `unorm-storage-image`
- `project.translate.wgsl-pointer-type-unsupported: WGSLPointerTypeUnsupportedError: WGSL target cannot represent CrossGL pointer type 'ptr<f32>': the spelling carries no address space, and WGSL pointers must name one (ptr<address-space, T>) (target wgsl) [project.translate.wgsl-pointer-type-unsupported]`: `pointer-local`
- `project.translate.wgsl-pointer-type-unsupported: WGSLPointerTypeUnsupportedError: WGSL target cannot represent CrossGL pointer type 'ptr<i32>': the spelling carries no address space, and WGSL pointers must name one (ptr<address-space, T>) (target wgsl) [project.translate.wgsl-pointer-type-unsupported]`: `buffer-in-helper`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image1D yet; split texture/sampler/storage bindings are required`: `storage-image-1d`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image2DArray yet; split texture/sampler/storage bindings are required`: `storage-image-array`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image3D yet; split texture/sampler/storage bindings are required`: `storage-image-3d`

</details>
