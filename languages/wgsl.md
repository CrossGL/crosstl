<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/wgsl.rst) describes how CrossTL translates WGSL.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 403 | 0 | 0 | 1 | 332 on vulkan-lavapipe | naga (30.0.1) |
| input | 100.0 % | 400 | 0 | 0 | 4 | 332 on vulkan-lavapipe | naga (30.0.1) |

## Feature lists

- [Tint's WGSL built-in function definitions](https://github.com/google/dawn/blob/ff7c6960f62d4249f967a78448918eec21705a86/src/tint/lang/wgsl/wgsl.def), pinned at `ff7c6960f62d`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 22/22 | 22/22 |
| control-flow | 56/56 | 56/56 |
| function | 145/145 | 142/142 |
| graphics | 16/16 | 16/16 |
| operator | 48/48 | 48/48 |
| resource | 57/57 | 57/57 |
| stage | 3/3 | 3/3 |
| type | 56/56 | 56/56 |

## Translation pairs

The cells of the translation matrix that translate WGSL straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| WGSL | CUDA | [97.2 %](../pairs/wgsl-to-cuda.md) | 207 | 6 | 0 | 3 | compile-only |
| WGSL | GLSL ES | [97.4 %](../pairs/wgsl-to-gles.md) | 342 | 7 | 2 | 5 | compile-only |
| WGSL | GLSL | [97.8 %](../pairs/wgsl-to-glsl.md) | 353 | 5 | 3 | 4 | yes |
| WGSL | HIP | [97.2 %](../pairs/wgsl-to-hip.md) | 210 | 6 | 0 | 3 | compile-only |
| WGSL | HLSL | [99.1 %](../pairs/wgsl-to-hlsl.md) | 344 | 0 | 3 | 4 | yes |
| WGSL | Metal | [100.0 %](../pairs/wgsl-to-metal.md) | 344 | 0 | 0 | 4 | compile-only |
| WGSL | Mojo | [92.6 %](../pairs/wgsl-to-mojo.md) | 200 | 13 | 3 | 2 | compile-only |
| WGSL | Rust | [93.3 %](../pairs/wgsl-to-rust.md) | 321 | 23 | 0 | 2 | yes |
| WGSL | Slang | [96.4 %](../pairs/wgsl-to-slang.md) | 346 | 2 | 11 | 2 | yes |
| WGSL | SPIR-V | [99.4 %](../pairs/wgsl-to-spirv.md) | 360 | 2 | 0 | 4 | yes |
| WGSL | WebGL | [80.0 %](../pairs/wgsl-to-webgl.md) | 12 | 3 | 0 | 0 | compile-only |
| CUDA | WGSL | [91.0 %](../pairs/cuda-to-wgsl.md) | 201 | 9 | 11 | 0 | compile-only |
| GLSL ES | WGSL | [98.6 %](../pairs/gles-to-wgsl.md) | 349 | 0 | 5 | 2 | compile-only |
| GLSL | WGSL | [97.5 %](../pairs/glsl-to-wgsl.md) | 357 | 1 | 8 | 1 | yes |
| HIP | WGSL | [91.0 %](../pairs/hip-to-wgsl.md) | 203 | 9 | 11 | 0 | compile-only |
| HLSL | WGSL | [87.0 %](../pairs/hlsl-to-wgsl.md) | 394 | 54 | 5 | 1 | yes |
| Metal | WGSL | [88.2 %](../pairs/metal-to-wgsl.md) | 315 | 28 | 14 | 1 | compile-only |
| Mojo | WGSL | [64.8 %](../pairs/mojo-to-wgsl.md) | 142 | 75 | 2 | 0 | compile-only |
| OpenCL C | WGSL | [84.6 %](../pairs/opencl-to-wgsl.md) | 192 | 30 | 5 | 0 | compile-only |
| Rust | WGSL | [52.5 %](../pairs/rust-to-wgsl.md) | 196 | 96 | 81 | 1 | yes |
| Slang | WGSL | [86.0 %](../pairs/slang-to-wgsl.md) | 325 | 18 | 35 | 1 | yes |
| SPIR-V | WGSL | [98.4 %](../pairs/spirv-to-wgsl.md) | 358 | 3 | 3 | 1 | yes |
| WebGL | WGSL | [98.8 %](../pairs/webgl-to-wgsl.md) | 253 | 0 | 3 | 0 | compile-only |

## Gaps

A diagnostic drops its line numbers and writes the probe's own variables as `<probe>`, and one that several features share writes each feature's name as `<feature>`.

### Output

Untested, by reason:

- `the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

### Input

Untested, by reason:

- `no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `no probe for parameters of type __modf_result_vec<N, T>`: `modf`
- `the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`
