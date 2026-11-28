<!-- Written from coverage.json by tools/coverage/docs.py when the Coverage workflow publishes. Do not edit. -->

# CUDA

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z (CrossTL 3.9.0) by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). The Coverage workflow writes this page from `coverage.json` each time it publishes; [COVERAGE.md](../COVERAGE.md) defines each outcome. Score = supported / (supported + refused + wrong).

The [backend guide](https://github.com/CrossGL/crosstl/blob/main/docs/source/backends/cuda.rst) describes how CrossTL translates CUDA.

| Direction | Score | Supported | Refused | Wrong | Untested | Executed | Reference tool |
|---|---:|---:|---:|---:|---:|---|---|
| output | 100.0 % | 718 | 0 | 0 | 0 | compile-only | nvcc: NVIDIA (R) Cuda compiler driver |
| input | 100.0 % | 718 | 0 | 0 | 0 | compile-only | nvcc: NVIDIA (R) Cuda compiler driver |

## Feature lists

- [LLVM 23.1.3 device-side CUDA Math API wrappers](https://github.com/llvm/llvm-project/blob/0d261d1ca552c95a8f007e061c787ac7132fbcbc/clang/lib/Headers/__clang_cuda_math.h), pinned at `0d261d1ca552`
- [LLVM 23.1.3 CUDA intrinsic and atomic wrappers](https://github.com/llvm/llvm-project/blob/0d261d1ca552c95a8f007e061c787ac7132fbcbc/clang/lib/Headers/__clang_cuda_device_functions.h), pinned at `0d261d1ca552`
- [LLVM 23.1.3 CUDA warp, cache-hinted load and store, and funnel shift intrinsics](https://github.com/llvm/llvm-project/blob/0d261d1ca552c95a8f007e061c787ac7132fbcbc/clang/lib/Headers/__clang_cuda_intrinsics.h), pinned at `0d261d1ca552`

## By category

Each cell counts the supported features of a category over its supported, refused and wrong ones.

| Category | output | input |
|---|---:|---:|
| compute | 18/18 | 18/18 |
| control-flow | 60/60 | 60/60 |
| function | 512/512 | 512/512 |
| operator | 34/34 | 34/34 |
| resource | 45/45 | 45/45 |
| stage | 1/1 | 1/1 |
| type | 48/48 | 48/48 |

## Translation pairs

The cells of the translation matrix that translate CUDA straight to another language or another language straight to it, measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z. Each cell measures the features of the source's input row that the target has too, and its page lists the gaps.

| Source | Target | Score | Supported | Refused | Wrong | Untested | Executed |
|---|---|---:|---:|---:|---:|---:|---|
| CUDA | GLSL ES | [96.7 %](../pairs/cuda-to-gles.md) | 232 | 2 | 6 | 2 | compile-only |
| CUDA | GLSL | [98.9 %](../pairs/cuda-to-glsl.md) | 273 | 3 | 0 | 0 | compile-only |
| CUDA | HIP | [100.0 %](../pairs/cuda-to-hip.md) | 532 | 0 | 0 | 0 | compile-only |
| CUDA | HLSL | [98.9 %](../pairs/cuda-to-hlsl.md) | 262 | 3 | 0 | 0 | compile-only |
| CUDA | Metal | [100.0 %](../pairs/cuda-to-metal.md) | 259 | 0 | 0 | 0 | compile-only |
| CUDA | Mojo | [88.8 %](../pairs/cuda-to-mojo.md) | 239 | 24 | 6 | 0 | compile-only |
| CUDA | Rust | [72.2 %](../pairs/cuda-to-rust.md) | 187 | 47 | 25 | 2 | compile-only |
| CUDA | Slang | [96.8 %](../pairs/cuda-to-slang.md) | 275 | 8 | 1 | 0 | compile-only |
| CUDA | SPIR-V | [96.9 %](../pairs/cuda-to-spirv.md) | 279 | 9 | 0 | 0 | compile-only |
| CUDA | WGSL | [91.0 %](../pairs/cuda-to-wgsl.md) | 201 | 9 | 11 | 0 | compile-only |
| GLSL ES | CUDA | [100.0 %](../pairs/gles-to-cuda.md) | 238 | 0 | 0 | 2 | compile-only |
| GLSL | CUDA | [100.0 %](../pairs/glsl-to-cuda.md) | 244 | 0 | 0 | 0 | compile-only |
| HIP | CUDA | [100.0 %](../pairs/hip-to-cuda.md) | 532 | 0 | 0 | 0 | compile-only |
| HLSL | CUDA | [98.8 %](../pairs/hlsl-to-cuda.md) | 237 | 3 | 0 | 0 | compile-only |
| Metal | CUDA | [94.6 %](../pairs/metal-to-cuda.md) | 261 | 15 | 0 | 0 | compile-only |
| Mojo | CUDA | [78.8 %](../pairs/mojo-to-cuda.md) | 175 | 46 | 1 | 0 | compile-only |
| OpenCL C | CUDA | [90.0 %](../pairs/opencl-to-cuda.md) | 217 | 22 | 2 | 0 | compile-only |
| Rust | CUDA | [81.0 %](../pairs/rust-to-cuda.md) | 204 | 28 | 20 | 1 | compile-only |
| Slang | CUDA | [97.6 %](../pairs/slang-to-cuda.md) | 247 | 6 | 0 | 0 | compile-only |
| SPIR-V | CUDA | [100.0 %](../pairs/spirv-to-cuda.md) | 249 | 0 | 0 | 2 | compile-only |
| WebGL | CUDA | [0.0 %](../pairs/webgl-to-cuda.md) | 0 | 177 | 0 | 0 | compile-only |
| WGSL | CUDA | [97.2 %](../pairs/wgsl-to-cuda.md) | 207 | 6 | 0 | 3 | compile-only |
