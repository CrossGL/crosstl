<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# HLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/directx.rst) describes how CrossTL translates HLSL.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 99.9 % | 830 | 0 | 1 | 0 | 530 on d3d12-warp, 360 on vulkan-lavapipe | dxc (libdxcompiler.so: 1.9(5191-d355aa83)(1.9.0.5191)) |
| input | 100.0 % | 831 | 0 | 0 | 0 | 531 on d3d12-warp, 361 on vulkan-lavapipe | dxc (libdxcompiler.so: 1.9(5191-d355aa83)(1.9.0.5191)) |

## Feature lists

- [DXC v1.9.2602.24 intrinsic table](https://github.com/microsoft/DirectXShaderCompiler/blob/d355aa8364d34df3f0822ba0de8d1dfc75ae6f48/utils/hct/gen_intrin_main.txt), pinned at `d355aa8364d3`
- [DXC v1.10.2605.37 (Shader Model 6.10 preview) intrinsic table](https://github.com/microsoft/DirectXShaderCompiler/blob/c4d8f4f99aa221da58cd540bf2099a2889632ab6/utils/hct/gen_intrin_main.txt), pinned at `c4d8f4f99aa2`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 13/13 | 13/13 |
| control-flow | 62/62 | 62/62 |
| function | 205/205 | 205/205 |
| graphics | 63/63 | 63/63 |
| method | 291/291 | 291/291 |
| operator | 50/50 | 50/50 |
| resource | 66/67 | 67/67 |
| stage | 14/14 | 14/14 |
| type | 66/66 | 66/66 |

## Translation pairs

The cells of the translation matrix that translate HLSL straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| HLSL | CUDA | [98.8 %](../pairs/hlsl-to-cuda.md) | 237 | 3 | 0 | 0 | compile-only |
| HLSL | GLSL ES | [95.2 %](../pairs/hlsl-to-gles.md) | 380 | 7 | 12 | 7 | compile-only |
| HLSL | GLSL | [99.8 %](../pairs/hlsl-to-glsl.md) | 443 | 0 | 1 | 0 | yes |
| HLSL | HIP | [98.8 %](../pairs/hlsl-to-hip.md) | 238 | 3 | 0 | 0 | compile-only |
| HLSL | Metal | [100.0 %](../pairs/hlsl-to-metal.md) | 382 | 0 | 0 | 0 | compile-only |
| HLSL | Mojo | [92.2 %](../pairs/hlsl-to-mojo.md) | 200 | 13 | 4 | 0 | compile-only |
| HLSL | Rust | [79.7 %](../pairs/hlsl-to-rust.md) | 440 | 99 | 13 | 0 | yes |
| HLSL | Slang | [96.7 %](../pairs/hlsl-to-slang.md) | 503 | 8 | 9 | 0 | yes |
| HLSL | SPIR-V | [97.6 %](../pairs/hlsl-to-spirv.md) | 440 | 8 | 3 | 0 | yes |
| HLSL | WebGL | [73.3 %](../pairs/hlsl-to-webgl.md) | 11 | 4 | 0 | 0 | compile-only |
| HLSL | WGSL | [87.0 %](../pairs/hlsl-to-wgsl.md) | 394 | 54 | 5 | 1 | yes |
| CUDA | HLSL | [98.9 %](../pairs/cuda-to-hlsl.md) | 262 | 3 | 0 | 0 | compile-only |
| GLSL ES | HLSL | [97.6 %](../pairs/gles-to-hlsl.md) | 373 | 8 | 1 | 7 | compile-only |
| GLSL | HLSL | [97.4 %](../pairs/glsl-to-hlsl.md) | 419 | 10 | 1 | 0 | yes |
| HIP | HLSL | [98.9 %](../pairs/hip-to-hlsl.md) | 263 | 3 | 0 | 0 | compile-only |
| Metal | HLSL | [95.6 %](../pairs/metal-to-hlsl.md) | 372 | 12 | 5 | 0 | compile-only |
| Mojo | HLSL | [73.3 %](../pairs/mojo-to-hlsl.md) | 159 | 57 | 1 | 0 | compile-only |
| Rust | HLSL | [48.0 %](../pairs/rust-to-hlsl.md) | 239 | 164 | 95 | 0 | yes |
| Slang | HLSL | [92.7 %](../pairs/slang-to-hlsl.md) | 493 | 11 | 28 | 0 | yes |
| SPIR-V | HLSL | [94.9 %](../pairs/spirv-to-hlsl.md) | 411 | 15 | 7 | 0 | yes |
| WebGL | HLSL | [99.6 %](../pairs/webgl-to-hlsl.md) | 256 | 0 | 1 | 0 | compile-only |
| WGSL | HLSL | [99.1 %](../pairs/wgsl-to-hlsl.md) | 344 | 0 | 3 | 4 | yes |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Output

Wrong, by cause ([open bug issues](https://github.com/CrossGL/crosstl/issues?q=is:issue%20is:open%20label:coverage%20label:bug%20label:backend:hlsl%20label:direction:output)):

- `on vulkan-lavapipe as SPIR-V from dxc, the translation binds other resources than the native probe`: `read-only-buffer`
