<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 97.9 % | 524 | 5 | 6 | 7 | compile-only | glslangValidator (Glslang Version: 11:16.6.0) |
| input | 99.3 % | 531 | 4 | 0 | 7 | compile-only | glslangValidator (Glslang Version: 11:16.6.0) |

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 28/28 | 28/28 |
| control-flow | 59/59 | 59/59 |
| function | 219/225 | 222/225 |
| graphics | 39/41 | 41/41 |
| operator | 50/51 | 51/51 |
| resource | 62/63 | 63/63 |
| stage | 6/6 | 6/6 |
| type | 61/62 | 61/62 |

## Translation pairs

The cells of the translation matrix that translate GLSL ES straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| GLSL ES | CUDA | [100.0 %](../pairs/gles-to-cuda.md) | 238 | 0 | 0 | 2 | compile-only |
| GLSL ES | GLSL | [100.0 %](../pairs/gles-to-glsl.md) | 535 | 0 | 0 | 7 | compile-only |
| GLSL ES | HIP | [100.0 %](../pairs/gles-to-hip.md) | 239 | 0 | 0 | 2 | compile-only |
| GLSL ES | HLSL | [97.6 %](../pairs/gles-to-hlsl.md) | 373 | 8 | 1 | 7 | compile-only |
| GLSL ES | Metal | [100.0 %](../pairs/gles-to-metal.md) | 383 | 0 | 0 | 6 | compile-only |
| GLSL ES | Mojo | [94.7 %](../pairs/gles-to-mojo.md) | 213 | 11 | 1 | 2 | compile-only |
| GLSL ES | Rust | [6.7 %](../pairs/gles-to-rust.md) | 27 | 376 | 1 | 5 | compile-only |
| GLSL ES | Slang | [96.8 %](../pairs/gles-to-slang.md) | 423 | 11 | 3 | 7 | compile-only |
| GLSL ES | SPIR-V | [99.4 %](../pairs/gles-to-spirv.md) | 464 | 3 | 0 | 7 | compile-only |
| GLSL ES | WebGL | [68.2 %](../pairs/gles-to-webgl.md) | 15 | 7 | 0 | 0 | compile-only |
| GLSL ES | WGSL | [98.6 %](../pairs/gles-to-wgsl.md) | 349 | 0 | 5 | 2 | compile-only |
| CUDA | GLSL ES | [96.7 %](../pairs/cuda-to-gles.md) | 232 | 2 | 6 | 2 | compile-only |
| GLSL | GLSL ES | [98.3 %](../pairs/glsl-to-gles.md) | 529 | 5 | 4 | 7 | compile-only |
| HIP | GLSL ES | [96.3 %](../pairs/hip-to-gles.md) | 232 | 2 | 7 | 2 | compile-only |
| HLSL | GLSL ES | [95.2 %](../pairs/hlsl-to-gles.md) | 380 | 7 | 12 | 7 | compile-only |
| Metal | GLSL ES | [94.7 %](../pairs/metal-to-gles.md) | 378 | 18 | 3 | 6 | compile-only |
| Mojo | GLSL ES | [63.3 %](../pairs/mojo-to-gles.md) | 145 | 70 | 14 | 2 | compile-only |
| OpenCL C | GLSL ES | [90.3 %](../pairs/opencl-to-gles.md) | 213 | 16 | 7 | 2 | compile-only |
| Rust | GLSL ES | [53.2 %](../pairs/rust-to-gles.md) | 230 | 174 | 28 | 5 | compile-only |
| Slang | GLSL ES | [93.9 %](../pairs/slang-to-gles.md) | 428 | 21 | 7 | 7 | compile-only |
| SPIR-V | GLSL ES | [93.0 %](../pairs/spirv-to-gles.md) | 422 | 27 | 5 | 7 | compile-only |
| WebGL | GLSL ES | [100.0 %](../pairs/webgl-to-gles.md) | 298 | 0 | 0 | 0 | compile-only |
| WGSL | GLSL ES | [97.4 %](../pairs/wgsl-to-gles.md) | 342 | 7 | 2 | 5 | compile-only |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Output

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:gles%20label:direction:output)):

- `'rgba32f' : format requires readonly or writeonly memory qualifier` (4): `imageLoad`, `imageSize`, `imageStore`, `storage-image`
- `'&' : wrong operand types: no operation '&' exists that takes a left-hand operand of type ' flat temp highp int SampleMaskIn' and a right operand of type ' const uint' (or there is no acceptable conversion)`: `sample-mask-input`
- `'+' : wrong operand types: no operation '+' exists that takes a left-hand operand of type ' temp highp float' and a right operand of type ' const int' (or there is no acceptable conversion)`: `implicit-conversion`

Refused, by diagnostic:

- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: GLSL version requirement: type 'double' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec2' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec3' does not exist in GLSL ES; the t` (3): `subgroupPartitionNV`, `subgroupPartitionedExclusiveAddNV`, `subgroupPartitionedExclusiveMulNV`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: the 'noperspective' qualifier requires GL_NV_shader_noperspective_interpolation; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `noperspective-interpolation`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'half' to 'half': half unsupported profile`: `half`

Untested, by reason:

- `the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `the probe itself is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the probe itself is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

### Input

Refused, by diagnostic:

- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: GLSL version requirement: type 'double' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec2' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec3' does not exist in GLSL ES; the t` (3): `subgroupPartitionNV`, `subgroupPartitionedExclusiveAddNV`, `subgroupPartitionedExclusiveMulNV`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'float16_t' to 'float16_t': half unsupported profile`: `half`

Untested, by reason:

- `the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `the probe itself is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the probe itself is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `the probe itself is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`
