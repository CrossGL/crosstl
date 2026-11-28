# CrossTL translation matrix: OpenCL C to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native OpenCL C probe whose feature WGSL has too straight to WGSL, and naga checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 84.6 %: 192 supported, 30 refused, 5 wrong, 0 untested.

## Gaps

<details>
<summary>35 gaps: 30 refused, 5 wrong</summary>

Wrong, by cause:

- `the type of 'lt' is expected to be 'vec2<i32>', but got 'vec2<bool>'` (2): `integer-vector-comparison`, `vector-comparison`
- `value nan cannot be represented as 'f32'` (2): `cos`, `sin`
- `the type of 'z' is expected to be 'vec2<i32>', but got 'vec2<bool>'`: `vector-conversion`

Refused, by diagnostic:

- `project.translate.wgsl-fixed-sampler-state-unsupported: FixedSamplerStateUnsupportedError: fixed sampler state sampler(...) cannot be expressed for wgsl: wgsl has no shader-declared sampler state (target wgsl) [project.translate.wgsl-fixed-sampler-state-unsupported]` (7): `sampled-texture`, `shared-sampler`, `texture-1d`, `texture-2d-array`, `texture-3d`, `texture-argument`, `texture-in-helper`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target requires storage image resource img to declare a representable image format such as layout(rgba8) or @rgba8` (7): `integer-storage-image`, `read-write-image`, `storage-image`, `storage-image-1d`, `storage-image-3d`, `storage-image-array`, `unorm-storage-image`
- `project.translate.wgsl-foreign-builtin-call: WGSLForeignBuiltinCallError: WGSL target cannot lower the call to <feature>(): <feature> is a built-in function of another language, and WGSL declares no built-in function of that name (target wgsl) [project.translate.wgsl-foreign-builtin-call]` (2): `atomic_load`, `atomic_store`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: absorb(q: ptr<Pair>)`: `aggregate-inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: advance(k: ptr<i32>)`: `loop-condition-call`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(a: ptr<f32>)`: `inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(k: ptr<i32>), leap(k: ptr<i32>)`: `conditional-side-effect`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: copy2(a: ptr<f32>), copy2(b: ptr<f32>), grow(a: ptr<f32>)`: `inout-argument-lvalues`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: deposit(iin: ptr<i32>), deposit(outp: ptr<f32>), fetch(iin: ptr<i32>), put(outp: ptr<f32>)`: `buffer-in-helper`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: exchange(tile: ptr<f32>), stash(tile: ptr<f32>)`: `shared-memory-in-helper`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: grow(x: ptr<f32>)`: `buffer-element-argument`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: inner(r: ptr<f32>), inner(s: ptr<f32>), outer(r: ptr<f32>), outer(s: ptr<f32>)`: `out-parameter-forwarding`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: limit(x: ptr<f32>)`: `void-return`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: slot(k: ptr<i32>)`: `indexed-compound-assignment`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: split(a: ptr<f32>), split(b: ptr<f32>)`: `out-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to wgsl because target lowering would produce invalid artifacts: unresolved non-void helper declarations without bodies: twice`: `forward-declaration`
- `project.translate.wgsl-pointer-type-unsupported: WGSLPointerTypeUnsupportedError: WGSL target cannot represent CrossGL pointer type 'ptr<f32>': the spelling carries no address space, and WGSL pointers must name one (ptr<address-space, T>) (target wgsl) [project.translate.wgsl-pointer-type-unsupported]`: `pointer-local`

</details>
