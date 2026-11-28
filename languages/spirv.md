<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# SPIR-V

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/spirv.rst) describes how CrossTL translates SPIR-V.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 565 | 0 | 0 | 3 | 414 on vulkan-lavapipe | spirv-val (SPIRV-Tools v2026.4 ef96ed763b43b59b33b31b362f09a02b729fa1c9) |
| input | 100.0 % | 567 | 0 | 0 | 1 | 417 on vulkan-lavapipe | spirv-val (SPIRV-Tools v2026.4 ef96ed763b43b59b33b31b362f09a02b729fa1c9) |

## Feature lists

- [SPIR-V GLSL.std.450 extended instruction grammar](https://github.com/KhronosGroup/SPIRV-Headers/blob/496543121ce6419f23d6fa5d7194ba66c36212d2/include/spirv/unified1/extinst.glsl.std.450.grammar.json), pinned at `496543121ce6`
- [SPIR-V core grammar (execution models, capabilities, storage classes, instructions)](https://github.com/KhronosGroup/SPIRV-Headers/blob/496543121ce6419f23d6fa5d7194ba66c36212d2/include/spirv/unified1/spirv.core.grammar.json), pinned at `496543121ce6`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 28/28 | 28/28 |
| control-flow | 62/62 | 62/62 |
| function | 199/199 | 201/201 |
| graphics | 73/73 | 73/73 |
| operator | 52/52 | 52/52 |
| resource | 67/67 | 67/67 |
| stage | 14/14 | 14/14 |
| type | 70/70 | 70/70 |

## Translation pairs

The cells of the translation matrix that translate SPIR-V straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| SPIR-V | CUDA | [100.0 %](../pairs/spirv-to-cuda.md) | 249 | 0 | 0 | 2 | compile-only |
| SPIR-V | GLSL ES | [93.0 %](../pairs/spirv-to-gles.md) | 422 | 27 | 5 | 7 | compile-only |
| SPIR-V | GLSL | [99.8 %](../pairs/spirv-to-glsl.md) | 509 | 1 | 0 | 0 | yes |
| SPIR-V | HIP | [100.0 %](../pairs/spirv-to-hip.md) | 250 | 0 | 0 | 2 | compile-only |
| SPIR-V | HLSL | [94.9 %](../pairs/spirv-to-hlsl.md) | 411 | 15 | 7 | 0 | yes |
| SPIR-V | Metal | [100.0 %](../pairs/spirv-to-metal.md) | 411 | 0 | 0 | 9 | compile-only |
| SPIR-V | Mojo | [75.0 %](../pairs/spirv-to-mojo.md) | 174 | 12 | 46 | 2 | compile-only |
| SPIR-V | Rust | [6.6 %](../pairs/spirv-to-rust.md) | 29 | 135 | 274 | 10 | yes |
| SPIR-V | Slang | [90.5 %](../pairs/spirv-to-slang.md) | 427 | 35 | 10 | 0 | yes |
| SPIR-V | WebGL | [75.0 %](../pairs/spirv-to-webgl.md) | 15 | 5 | 0 | 0 | compile-only |
| SPIR-V | WGSL | [98.4 %](../pairs/spirv-to-wgsl.md) | 358 | 3 | 3 | 1 | yes |
| CUDA | SPIR-V | [96.9 %](../pairs/cuda-to-spirv.md) | 279 | 9 | 0 | 0 | compile-only |
| GLSL ES | SPIR-V | [99.4 %](../pairs/gles-to-spirv.md) | 464 | 3 | 0 | 7 | compile-only |
| GLSL | SPIR-V | [99.2 %](../pairs/glsl-to-spirv.md) | 522 | 4 | 0 | 0 | yes |
| HIP | SPIR-V | [96.5 %](../pairs/hip-to-spirv.md) | 279 | 10 | 0 | 0 | compile-only |
| HLSL | SPIR-V | [97.6 %](../pairs/hlsl-to-spirv.md) | 440 | 8 | 3 | 0 | yes |
| Metal | SPIR-V | [93.4 %](../pairs/metal-to-spirv.md) | 408 | 26 | 3 | 0 | compile-only |
| Mojo | SPIR-V | [71.4 %](../pairs/mojo-to-spirv.md) | 170 | 67 | 1 | 0 | compile-only |
| OpenCL C | SPIR-V | [94.4 %](../pairs/opencl-to-spirv.md) | 234 | 14 | 0 | 0 | compile-only |
| Rust | SPIR-V | [55.2 %](../pairs/rust-to-spirv.md) | 262 | 203 | 10 | 0 | yes |
| Slang | SPIR-V | [91.0 %](../pairs/slang-to-spirv.md) | 443 | 12 | 32 | 0 | yes |
| WebGL | SPIR-V | [99.7 %](../pairs/webgl-to-spirv.md) | 296 | 1 | 0 | 0 | compile-only |
| WGSL | SPIR-V | [99.4 %](../pairs/wgsl-to-spirv.md) | 360 | 2 | 0 | 4 | yes |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Output

Untested, by reason:

- `alternative encoding of Frexp, chosen by the compiler`: `FrexpStruct`
- `alternative encoding of Modf, chosen by the compiler`: `ModfStruct`
- `reserved by the GLSL.std.450 specification, not a valid instruction`: `IMix`

### Input

Untested, by reason:

- `reserved by the GLSL.std.450 specification, not a valid instruction`: `IMix`
