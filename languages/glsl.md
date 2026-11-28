<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# GLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/glsl.rst) describes how CrossTL translates GLSL.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 604 | 0 | 0 | 0 | 391 on opengl-llvmpipe, 80 on vulkan-lavapipe | glslangValidator (Glslang Version: 11:16.6.0) |
| input | 100.0 % | 604 | 0 | 0 | 0 | 391 on opengl-llvmpipe, 80 on vulkan-lavapipe | glslangValidator (Glslang Version: 11:16.6.0) |

## Feature lists

- [GLSL 4.60 specification, chapter 8 (built-in functions)](https://github.com/KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/chapters/builtinfunctions.adoc), pinned at `0099bf83b028`
- [GL_EXT_shader_quad_control (quad votes)](https://github.com/KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/extensions/ext/GLSL_EXT_shader_quad_control.txt), pinned at `0099bf83b028`
- [GL_KHR_shader_subgroup (section 8.18, shader invocation group functions)](https://github.com/KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/extensions/khr/GL_KHR_shader_subgroup.txt), pinned at `0099bf83b028`
- [GL_NV_shader_subgroup_partitioned (partitioned subgroup operations)](https://github.com/KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/extensions/nv/GL_NV_shader_subgroup_partitioned.txt), pinned at `0099bf83b028`
- [GL_KHR_shader_subgroup_rotate (subgroup rotations)](https://github.com/KhronosGroup/GLSL/blob/0099bf83b0288469917fc2e2972cb0a2b74d6711/extensions/khr/GL_KHR_shader_subgroup_rotate.txt), pinned at `0099bf83b028`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 28/28 | 28/28 |
| control-flow | 59/59 | 59/59 |
| function | 251/251 | 251/251 |
| graphics | 69/69 | 69/69 |
| operator | 51/51 | 51/51 |
| resource | 67/67 | 67/67 |
| stage | 14/14 | 14/14 |
| type | 65/65 | 65/65 |

## Translation pairs

The cells of the translation matrix that translate GLSL straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| GLSL | CUDA | [100.0 %](../pairs/glsl-to-cuda.md) | 244 | 0 | 0 | 0 | compile-only |
| GLSL | GLSL ES | [98.3 %](../pairs/glsl-to-gles.md) | 529 | 5 | 4 | 7 | compile-only |
| GLSL | HIP | [100.0 %](../pairs/glsl-to-hip.md) | 245 | 0 | 0 | 0 | compile-only |
| GLSL | HLSL | [97.4 %](../pairs/glsl-to-hlsl.md) | 419 | 10 | 1 | 0 | yes |
| GLSL | Metal | [100.0 %](../pairs/glsl-to-metal.md) | 414 | 0 | 0 | 0 | compile-only |
| GLSL | Mojo | [94.7 %](../pairs/glsl-to-mojo.md) | 216 | 11 | 1 | 0 | compile-only |
| GLSL | Rust | [8.4 %](../pairs/glsl-to-rust.md) | 38 | 416 | 1 | 0 | yes |
| GLSL | Slang | [95.9 %](../pairs/glsl-to-slang.md) | 470 | 17 | 3 | 0 | yes |
| GLSL | SPIR-V | [99.2 %](../pairs/glsl-to-spirv.md) | 522 | 4 | 0 | 0 | yes |
| GLSL | WebGL | [68.2 %](../pairs/glsl-to-webgl.md) | 15 | 7 | 0 | 0 | compile-only |
| GLSL | WGSL | [97.5 %](../pairs/glsl-to-wgsl.md) | 357 | 1 | 8 | 1 | yes |
| CUDA | GLSL | [98.9 %](../pairs/cuda-to-glsl.md) | 273 | 3 | 0 | 0 | compile-only |
| GLSL ES | GLSL | [100.0 %](../pairs/gles-to-glsl.md) | 535 | 0 | 0 | 7 | compile-only |
| HIP | GLSL | [98.9 %](../pairs/hip-to-glsl.md) | 274 | 3 | 0 | 0 | compile-only |
| HLSL | GLSL | [99.8 %](../pairs/hlsl-to-glsl.md) | 443 | 0 | 1 | 0 | yes |
| Metal | GLSL | [97.4 %](../pairs/metal-to-glsl.md) | 417 | 8 | 3 | 0 | compile-only |
| Mojo | GLSL | [65.9 %](../pairs/mojo-to-glsl.md) | 153 | 64 | 15 | 0 | compile-only |
| OpenCL C | GLSL | [94.2 %](../pairs/opencl-to-glsl.md) | 228 | 14 | 0 | 0 | compile-only |
| Rust | GLSL | [52.6 %](../pairs/rust-to-glsl.md) | 253 | 198 | 30 | 0 | yes |
| Slang | GLSL | [96.4 %](../pairs/slang-to-glsl.md) | 488 | 8 | 10 | 0 | yes |
| SPIR-V | GLSL | [99.8 %](../pairs/spirv-to-glsl.md) | 509 | 1 | 0 | 0 | yes |
| WebGL | GLSL | [100.0 %](../pairs/webgl-to-glsl.md) | 298 | 0 | 0 | 0 | compile-only |
| WGSL | GLSL | [97.8 %](../pairs/wgsl-to-glsl.md) | 353 | 5 | 3 | 4 | yes |
