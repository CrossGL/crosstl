<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# HIP

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/hip.rst) describes how CrossTL translates HIP.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 567 | 0 | 0 | 0 | compile-only | hipcc (HIP version: 5.7.31921-0) |
| input | 100.0 % | 567 | 0 | 0 | 0 | compile-only | hipcc (HIP version: 5.7.31921-0) |

## Feature lists

- [LLVM 17.0.6 device-side HIP math functions and intrinsics](https://github.com/llvm/llvm-project/blob/6009708b4367171ccdbf4b5905cb6a803753fe18/clang/lib/Headers/__clang_hip_math.h), pinned at `6009708b4367`
- [ROCm 5.7.1 HIP atomic functions](https://github.com/ROCm/clr/blob/1949b1621a802ffb1492616adbae6154bfbe64ef/hipamd/include/hip/amd_detail/amd_hip_atomic.h), pinned at `1949b1621a80`
- [ROCm 5.7.1 HIP device functions (integer intrinsics, conversions, barriers)](https://github.com/ROCm/clr/blob/1949b1621a802ffb1492616adbae6154bfbe64ef/hipamd/include/hip/amd_detail/amd_device_functions.h), pinned at `1949b1621a80`
- [ROCm 5.7.1 HIP warp shuffles and lane permutes](https://github.com/ROCm/clr/blob/1949b1621a802ffb1492616adbae6154bfbe64ef/hipamd/include/hip/amd_detail/amd_warp_functions.h), pinned at `1949b1621a80`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 18/18 | 18/18 |
| control-flow | 60/60 | 60/60 |
| function | 361/361 | 361/361 |
| operator | 34/34 | 34/34 |
| resource | 45/45 | 45/45 |
| stage | 1/1 | 1/1 |
| type | 48/48 | 48/48 |

## Translation pairs

The cells of the translation matrix that translate HIP straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| HIP | CUDA | [100.0 %](../pairs/hip-to-cuda.md) | 532 | 0 | 0 | 0 | compile-only |
| HIP | GLSL ES | [96.3 %](../pairs/hip-to-gles.md) | 232 | 2 | 7 | 2 | compile-only |
| HIP | GLSL | [98.9 %](../pairs/hip-to-glsl.md) | 274 | 3 | 0 | 0 | compile-only |
| HIP | HLSL | [98.9 %](../pairs/hip-to-hlsl.md) | 263 | 3 | 0 | 0 | compile-only |
| HIP | Metal | [99.6 %](../pairs/hip-to-metal.md) | 259 | 0 | 1 | 0 | compile-only |
| HIP | Mojo | [87.1 %](../pairs/hip-to-mojo.md) | 237 | 26 | 9 | 0 | compile-only |
| HIP | Rust | [70.7 %](../pairs/hip-to-rust.md) | 186 | 52 | 25 | 2 | compile-only |
| HIP | Slang | [96.8 %](../pairs/hip-to-slang.md) | 276 | 8 | 1 | 0 | compile-only |
| HIP | SPIR-V | [96.5 %](../pairs/hip-to-spirv.md) | 279 | 10 | 0 | 0 | compile-only |
| HIP | WGSL | [91.0 %](../pairs/hip-to-wgsl.md) | 203 | 9 | 11 | 0 | compile-only |
| CUDA | HIP | [100.0 %](../pairs/cuda-to-hip.md) | 532 | 0 | 0 | 0 | compile-only |
| GLSL ES | HIP | [100.0 %](../pairs/gles-to-hip.md) | 239 | 0 | 0 | 2 | compile-only |
| GLSL | HIP | [100.0 %](../pairs/glsl-to-hip.md) | 245 | 0 | 0 | 0 | compile-only |
| HLSL | HIP | [98.8 %](../pairs/hlsl-to-hip.md) | 238 | 3 | 0 | 0 | compile-only |
| Metal | HIP | [94.6 %](../pairs/metal-to-hip.md) | 262 | 15 | 0 | 0 | compile-only |
| Mojo | HIP | [78.1 %](../pairs/mojo-to-hip.md) | 175 | 47 | 2 | 0 | compile-only |
| OpenCL C | HIP | [89.7 %](../pairs/opencl-to-hip.md) | 218 | 22 | 3 | 0 | compile-only |
| Rust | HIP | [80.0 %](../pairs/rust-to-hip.md) | 204 | 30 | 21 | 1 | compile-only |
| Slang | HIP | [97.3 %](../pairs/slang-to-hip.md) | 248 | 6 | 1 | 0 | compile-only |
| SPIR-V | HIP | [100.0 %](../pairs/spirv-to-hip.md) | 250 | 0 | 0 | 2 | compile-only |
| WebGL | HIP | [0.0 %](../pairs/webgl-to-hip.md) | 0 | 177 | 0 | 0 | compile-only |
| WGSL | HIP | [97.2 %](../pairs/wgsl-to-hip.md) | 210 | 6 | 0 | 3 | compile-only |
