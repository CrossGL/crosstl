<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# Metal

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/metal.rst) describes how CrossTL translates Metal.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 545 | 0 | 0 | 0 | 480 on metal-apple | metal (Apple metal version 32023.921 (metalfe-32023.921.6)) |
| input | 100.0 % | 545 | 0 | 0 | 0 | 481 on metal-apple | metal (Apple metal version 32023.921 (metalfe-32023.921.6)) |

## Feature lists

- [Metal Shading Language 4.1 standard library, from Apple metal version 32023.918 (metalfe-32023.918.1)](https://github.com/CrossGL/crosstl/blob/8f9a2e88b1fda43915bc783f1c4a52557476f9a9/tools/coverage/features/metal.txt), kept in this repository

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 25/25 | 25/25 |
| control-flow | 62/62 | 62/62 |
| function | 224/224 | 224/224 |
| graphics | 45/45 | 45/45 |
| operator | 51/51 | 51/51 |
| resource | 64/64 | 64/64 |
| stage | 7/7 | 7/7 |
| type | 67/67 | 67/67 |

## Translation pairs

The cells of the translation matrix that translate Metal straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| Metal | CUDA | [94.6 %](../pairs/metal-to-cuda.md) | 261 | 15 | 0 | 0 | compile-only |
| Metal | GLSL ES | [94.7 %](../pairs/metal-to-gles.md) | 378 | 18 | 3 | 6 | compile-only |
| Metal | GLSL | [97.4 %](../pairs/metal-to-glsl.md) | 417 | 8 | 3 | 0 | compile-only |
| Metal | HIP | [94.6 %](../pairs/metal-to-hip.md) | 262 | 15 | 0 | 0 | compile-only |
| Metal | HLSL | [95.6 %](../pairs/metal-to-hlsl.md) | 372 | 12 | 5 | 0 | compile-only |
| Metal | Mojo | [71.1 %](../pairs/metal-to-mojo.md) | 182 | 70 | 4 | 0 | compile-only |
| Metal | Rust | [69.5 %](../pairs/metal-to-rust.md) | 276 | 112 | 9 | 2 | compile-only |
| Metal | Slang | [92.5 %](../pairs/metal-to-slang.md) | 395 | 27 | 5 | 0 | compile-only |
| Metal | SPIR-V | [93.4 %](../pairs/metal-to-spirv.md) | 408 | 26 | 3 | 0 | compile-only |
| Metal | WebGL | [83.3 %](../pairs/metal-to-webgl.md) | 15 | 3 | 0 | 0 | compile-only |
| Metal | WGSL | [88.2 %](../pairs/metal-to-wgsl.md) | 315 | 28 | 14 | 1 | compile-only |
| CUDA | Metal | [100.0 %](../pairs/cuda-to-metal.md) | 259 | 0 | 0 | 0 | compile-only |
| GLSL ES | Metal | [100.0 %](../pairs/gles-to-metal.md) | 383 | 0 | 0 | 6 | compile-only |
| GLSL | Metal | [100.0 %](../pairs/glsl-to-metal.md) | 414 | 0 | 0 | 0 | compile-only |
| HIP | Metal | [99.6 %](../pairs/hip-to-metal.md) | 259 | 0 | 1 | 0 | compile-only |
| HLSL | Metal | [100.0 %](../pairs/hlsl-to-metal.md) | 382 | 0 | 0 | 0 | compile-only |
| Mojo | Metal | [70.4 %](../pairs/mojo-to-metal.md) | 174 | 72 | 1 | 0 | compile-only |
| OpenCL C | Metal | [93.7 %](../pairs/opencl-to-metal.md) | 251 | 15 | 2 | 0 | compile-only |
| Rust | Metal | [59.4 %](../pairs/rust-to-metal.md) | 243 | 106 | 60 | 2 | compile-only |
| Slang | Metal | [98.6 %](../pairs/slang-to-metal.md) | 426 | 5 | 1 | 0 | compile-only |
| SPIR-V | Metal | [100.0 %](../pairs/spirv-to-metal.md) | 411 | 0 | 0 | 9 | compile-only |
| WebGL | Metal | [99.6 %](../pairs/webgl-to-metal.md) | 260 | 0 | 1 | 0 | compile-only |
| WGSL | Metal | [100.0 %](../pairs/wgsl-to-metal.md) | 344 | 0 | 0 | 4 | compile-only |
