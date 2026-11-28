# CrossTL translation matrix: OpenCL C to Rust

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native OpenCL C probe whose feature Rust has too straight to Rust, and rust-gpu checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 78.0 %: 188 supported, 48 refused, 5 wrong, 1 untested.

## Gaps

<details>
<summary>53 gaps: 48 refused, 5 wrong</summary>

Wrong, by cause:

- `error[E0308]: mismatched types` (3): `constant-struct`, `struct-constructor`, `zero-initializer`
- `error[E0425]: cannot find type 'ptr' in this scope`: `pointer-local`
- `error[E0600]: cannot apply unary operator '-' to type 'spirv_std::glam::BVec2'`: `vector-conversion`

Refused, by diagnostic:

- `project.translate.rust-fixed-sampler-state-unsupported: FixedSamplerStateUnsupportedError: fixed sampler state sampler(...) cannot be expressed for rust: rust has no shader-declared sampler state (target rust) [project.translate.rust-fixed-sampler-state-unsupported]` (8): `sampled-texture`, `shared-sampler`, `texture-1d`, `texture-1d-array`, `texture-2d-array`, `texture-3d`, `texture-argument`, `texture-in-helper`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'probe_Args': type 'f32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]` (5): `uniform-cbuffer-vector`, `uniform-members`, `uniform-std140-array`, `uniform-std140-vector`, `uniform-vector-packing`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: '<feature>' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]` (3): `distance`, `nan`, `select`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: storage image 'img' (image2D) has no format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities (target rust) [project.translate.rust-lowering-unsupported]` (3): `read-write-image`, `storage-image`, `unorm-storage-image`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: atomic: <feature>'s location is not in a storage buffer or workgroup memory, so no other invocation shares it (target rust) [project.translate.rust-lowering-unsupported]` (2): `atomic_load`, `atomic_store`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'probe_Args': type 'u32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]` (2): `uniform-bool`, `uniform-unsigned`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N constructs: function: 'any' has no rust-gpu lowering for these operands; function: 'all' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]` (2): `integer-vector-comparison`, `vector-comparison`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: absorb(q: ptr<Pair>)`: `aggregate-inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: advance(k: ptr<i32>)`: `loop-condition-call`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(a: ptr<f32>)`: `inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(k: ptr<i32>), leap(k: ptr<i32>)`: `conditional-side-effect`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: copy2(a: ptr<f32>), copy2(b: ptr<f32>), grow(a: ptr<f32>)`: `inout-argument-lvalues`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: deposit(iin: ptr<i32>), deposit(outp: ptr<f32>), fetch(iin: ptr<i32>), put(outp: ptr<f32>)`: `buffer-in-helper`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: exchange(tile: ptr<f32>), stash(tile: ptr<f32>)`: `shared-memory-in-helper`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: grow(x: ptr<f32>)`: `buffer-element-argument`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: inner(r: ptr<f32>), inner(s: ptr<f32>), outer(r: ptr<f32>), outer(s: ptr<f32>)`: `out-parameter-forwarding`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: limit(x: ptr<f32>)`: `void-return`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: slot(k: ptr<i32>)`: `indexed-compound-assignment`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: split(a: ptr<f32>), split(b: ptr<f32>)`: `out-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to rust because target lowering would produce invalid artifacts: unresolved non-void helper declarations without bodies: twice`: `forward-declaration`
- `project.translate.opencl-printf-value-unsupported: OpenCLPrintfValueUnsupportedError: OpenCL C printf returns an int (N, or -N on failure), and CrossGL printf, the Vulkan debug print, returns nothing, so its result cannot be read [project.translate.opencl-printf-value-unsupported]`: `printf`
- `project.translate.opencl-image-read-unsupported: OpenCLImageReadUnsupportedError: read_imagef on tex: image1d_buffer_t has no CrossGL read (a depth image read returns one float, and a buffer image is not a texture) [project.translate.opencl-image-read-unsupported]`: `texel-buffer`
- `project.translate.opencl-record-semantics-unsupported: OpenCLRecordSemanticError: OpenCL union overlapping storage access from 'f' to 'u' is unsupported for record 'Bits' in object 'b'`: `union`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: built-in: 'gl_WorkGroupSize' is used by name, and the rust-gpu output binds no entry-point parameter for it (target rust) [project.translate.rust-lowering-unsupported]`: `subgroup-size`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'probe_Args': type 'i32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]`: `uniform-integer`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: storage image 'img' (image1D) has no format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities (target rust) [project.translate.rust-lowering-unsupported]`: `storage-image-1d`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: storage image 'img' (image2DArray) has no format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities (target rust) [project.translate.rust-lowering-unsupported]`: `storage-image-array`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: storage image 'img' (image3D) has no format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities (target rust) [project.translate.rust-lowering-unsupported]`: `storage-image-3d`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: storage image 'img' (imageBuffer) has no format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities (target rust) [project.translate.rust-lowering-unsupported]`: `storage-texel-buffer`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: storage image 'img' (uimage2D) has no format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities (target rust) [project.translate.rust-lowering-unsupported]`: `integer-storage-image`

</details>

<details>
<summary>1 untested: function 1</summary>

- `the target's own probe is rejected: error: Hypot not supported yet`: `hypot`

</details>
