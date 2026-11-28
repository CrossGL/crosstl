# CrossTL translation matrix: OpenCL C to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native OpenCL C probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 90.3 %: 213 supported, 16 refused, 7 wrong, 2 untested.

## Gaps

<details>
<summary>23 gaps: 16 refused, 7 wrong</summary>

Wrong, by cause:

- `'rgba32f' : format requires readonly or writeonly memory qualifier` (5): `read-write-image`, `storage-image`, `storage-image-3d`, `storage-image-array`, `unorm-storage-image`
- `'+' : wrong operand types: no operation '+' exists that takes a left-hand operand of type ' temp highp float' and a right operand of type ' const int' (or there is no acceptable conversion)`: `implicit-conversion`
- `'mod' : undeclared identifier`: `reserved-global-name`

Refused, by diagnostic:

- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: absorb(q: ptr<Pair>)`: `aggregate-inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: advance(k: ptr<i32>)`: `loop-condition-call`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(a: ptr<f32>)`: `inout-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: bump(k: ptr<i32>), leap(k: ptr<i32>)`: `conditional-side-effect`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: copy2(a: ptr<f32>), copy2(b: ptr<f32>), grow(a: ptr<f32>)`: `inout-argument-lvalues`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: deposit(iin: ptr<i32>), deposit(outp: ptr<f32>), fetch(iin: ptr<i32>), put(outp: ptr<f32>)`: `buffer-in-helper`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: exchange(tile: ptr<f32>), stash(tile: ptr<f32>)`: `shared-memory-in-helper`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: grow(x: ptr<f32>)`: `buffer-element-argument`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: inner(r: ptr<f32>), inner(s: ptr<f32>), outer(r: ptr<f32>), outer(s: ptr<f32>)`: `out-parameter-forwarding`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: limit(x: ptr<f32>)`: `void-return`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: slot(k: ptr<i32>)`: `indexed-compound-assignment`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: OpenCL pointer helper parameters are not representable in target codegen: split(a: ptr<f32>), split(b: ptr<f32>)`: `out-parameter`
- `opencl.target.unsupported: OpenCLTargetUnsupportedError: opencl.target.unsupported: cannot lower OpenCL source to opengl because target lowering would produce invalid artifacts: unresolved non-void helper declarations without bodies: twice`: `forward-declaration`
- `project.translate.opencl-image-read-unsupported: OpenCLImageReadUnsupportedError: read_imagef on tex: image1d_buffer_t has no CrossGL read (a depth image read returns one float, and a buffer image is not a texture) [project.translate.opencl-image-read-unsupported]`: `texel-buffer`
- `project.translate.opengl-fmod-unrepresentable: OpenGLFloatingRemainderError: OpenGL cannot preserve floating remainder for ('f32', 'float'): unsupported-profile`: `float-remainder`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'f16' to 'f16': half unsupported profile`: `half`

</details>

<details>
<summary>2 untested: type 2</summary>

- `the target's own probe is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`

</details>
