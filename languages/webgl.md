<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# WebGL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/webgl.rst) describes how CrossTL translates WebGL.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 298 | 0 | 0 | 0 | compile-only | glslangValidator (Glslang Version: 11:16.6.0) |
| input | 100.0 % | 298 | 0 | 0 | 0 | compile-only | glslangValidator (Glslang Version: 11:16.6.0) |

## Feature lists

- [ANGLE's GLSL ES built-in function table](https://github.com/google/angle/blob/1ff891807e391d842c33d552ea26ab26b5341ef4/src/compiler/translator/builtin_function_declarations.txt), pinned at `1ff891807e39`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| control-flow | 58/58 | 58/58 |
| function | 89/89 | 89/89 |
| graphics | 12/12 | 12/12 |
| operator | 50/50 | 50/50 |
| resource | 32/32 | 32/32 |
| stage | 2/2 | 2/2 |
| type | 55/55 | 55/55 |

## Translation pairs

The cells of the translation matrix that translate WebGL straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| WebGL | CUDA | [0.0 %](../pairs/webgl-to-cuda.md) | 0 | 177 | 0 | 0 | compile-only |
| WebGL | GLSL ES | [100.0 %](../pairs/webgl-to-gles.md) | 298 | 0 | 0 | 0 | compile-only |
| WebGL | GLSL | [100.0 %](../pairs/webgl-to-glsl.md) | 298 | 0 | 0 | 0 | compile-only |
| WebGL | HIP | [0.0 %](../pairs/webgl-to-hip.md) | 0 | 177 | 0 | 0 | compile-only |
| WebGL | HLSL | [99.6 %](../pairs/webgl-to-hlsl.md) | 256 | 0 | 1 | 0 | compile-only |
| WebGL | Metal | [99.6 %](../pairs/webgl-to-metal.md) | 260 | 0 | 1 | 0 | compile-only |
| WebGL | Mojo | [97.1 %](../pairs/webgl-to-mojo.md) | 165 | 5 | 0 | 0 | compile-only |
| WebGL | Rust | [96.8 %](../pairs/webgl-to-rust.md) | 245 | 7 | 1 | 0 | compile-only |
| WebGL | Slang | [99.6 %](../pairs/webgl-to-slang.md) | 265 | 0 | 1 | 0 | compile-only |
| WebGL | SPIR-V | [99.7 %](../pairs/webgl-to-spirv.md) | 296 | 1 | 0 | 0 | compile-only |
| WebGL | WGSL | [98.8 %](../pairs/webgl-to-wgsl.md) | 253 | 0 | 3 | 0 | compile-only |
| GLSL ES | WebGL | [68.2 %](../pairs/gles-to-webgl.md) | 15 | 7 | 0 | 0 | compile-only |
| GLSL | WebGL | [68.2 %](../pairs/glsl-to-webgl.md) | 15 | 7 | 0 | 0 | compile-only |
| HLSL | WebGL | [73.3 %](../pairs/hlsl-to-webgl.md) | 11 | 4 | 0 | 0 | compile-only |
| Metal | WebGL | [83.3 %](../pairs/metal-to-webgl.md) | 15 | 3 | 0 | 0 | compile-only |
| Rust | WebGL | [55.6 %](../pairs/rust-to-webgl.md) | 10 | 7 | 1 | 0 | compile-only |
| Slang | WebGL | [64.7 %](../pairs/slang-to-webgl.md) | 11 | 5 | 1 | 0 | compile-only |
| SPIR-V | WebGL | [75.0 %](../pairs/spirv-to-webgl.md) | 15 | 5 | 0 | 0 | compile-only |
| WGSL | WebGL | [80.0 %](../pairs/wgsl-to-webgl.md) | 12 | 3 | 0 | 0 | compile-only |
