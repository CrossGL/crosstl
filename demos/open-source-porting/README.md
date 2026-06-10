# Open-Source Project Porting Demo

This directory contains small, pinned repository slices that exercise the
project translation pipeline against real open-source shader and GPU sources.
The cases are intentionally source-focused: CrossTL translates shader or kernel
artifacts, records reports, and leaves host runtime integration, resource
binding, and build-system migration as manual follow-up work.

## Running the demo

Regenerate every case in a temporary copy and compare the generated artifacts
with the checked-in references:

```bash
python demos/open-source-porting/run_demo.py --check
```

Refresh the checked-in reference artifacts after an intentional translator
change:

```bash
python demos/open-source-porting/run_demo.py --update
```

Run a platform-target subset with validation toolchain smoke checks:

```bash
python demos/open-source-porting/run_demo.py \
  --check \
  --run-toolchains \
  --require-toolchain-runs \
  --case directx-graphics-samples-hello-triangle \
  --target opengl \
  --target vulkan
```

The demo workflow in `.github/workflows/demo.yml` runs the reproducibility
check on Linux, macOS, and Windows. It also runs target-specific smoke checks:
OpenGL and Vulkan on Linux, Metal on macOS, and DirectX on Windows.

## Demo cases

| Case | Upstream | License | Source backend | Checked targets | Notes |
| --- | --- | --- | --- | --- | --- |
| `directx-graphics-samples-hello-triangle` | `microsoft/DirectX-Graphics-Samples` at `31ae3c91160d8634264004cdaf4e41a99c41243e` | MIT | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Hello Triangle shader file without source edits. |
| `lonelydevil-vulkan-tutorial-triangle` | `lonelydevil/vulkan-tutorial-C-implementation` at `780ff146a6eccd7064a10e86363f3c2f7323825d` | MIT | GLSL | CrossGL, OpenGL, DirectX, Vulkan | Uses the upstream triangle shader pair unchanged. Metal vertex-index lowering is tracked in issue #803. |
| `vulkan-samples-dynamic-line-grid` | `KhronosGroup/Vulkan-Samples` at `ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a` | Apache-2.0 | GLSL | CrossGL, Metal, DirectX, Vulkan | Uses the reduced fragment shader already covered by backend fixture provenance. OpenGL smoke validation for this case is tracked in issue #745. |
| `apple-modern-rendering-mesh-viewdir` | `donaldwuid/apple_metal_sample_code` at `0bc50e5b3670b3169855ab260e8da5ff07b53749` | MIT | Metal | CrossGL, Metal, DirectX, Vulkan | Uses a reduced shader slice that keeps the relevant vertex-stage type conversion. OpenGL output is tracked in issue #746. |
| `metal-performance-testing-matmul` | `bkvogel/metal_performance_testing` at `b467b4b1dee0f7d9d43bda13856306ca3f1baea5` | BSD-style | Metal | CrossGL, Metal, Vulkan | Uses the upstream Metal kernel and its shared parameter header. DirectX constant-parameter lowering is tracked in issue #755. |
| `nvidia-cuda-samples-vector-add` | `NVIDIA/cuda-samples` at `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` | BSD-style with CUDA EULA reference | CUDA | CrossGL, Metal, Vulkan | Uses the upstream NVRTC vectorAdd kernel unchanged. Host launch and memory-management integration remain outside the demo scope. |
| `opencl-sdk-saxpy` | `KhronosGroup/OpenCL-SDK` at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` | Apache-2.0 | OpenCL | CrossGL, Metal, Vulkan | Uses the upstream SAXPY compute kernel unchanged. OpenGL index-cast lowering is tracked in issue #768. |
| `raylib-base-fragment` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream base fragment shader unchanged. |
| `raylib-base-vertex` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream base vertex shader unchanged. |
| `raylib-lighting-shader-pair` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream lighting vertex and fragment shaders unchanged. |
| `rust-gpu-graphics-stage-inputs` | `Rust-GPU/rust-gpu` at `36e3348cdc2f824afec64b3b5af5d369d98a4c0d` | Apache-2.0 OR MIT | Rust-GPU | CrossGL, OpenGL, Metal, Vulkan | Uses a reduced graphics shader slice that keeps the plain vertex input and fragment color path. |
| `rust-gpu-vulkan-examples-triangle-overlay` | `Rust-GPU/VulkanShaderExamples` at `b29a37eb46802b5ea6882af4808d6887fc184581` | MIT | Rust-GPU | CrossGL, Metal, Vulkan | Uses the upstream conservative raster triangle-overlay shader unchanged. |
| `sascha-willems-vulkan-conservative-triangle` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream conservative raster triangle vertex shader without semantic edits. DirectX semantic lowering remains outside this checked target subset. |
| `slang-hello-world-compute` | `shader-slang/slang` at `29e69b0bf626f87500be73a7fb3764db25658c66` | Apache-2.0 WITH LLVM-exception | Slang | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream compute shader unchanged. |
| `spirv-tools-basic-src` | `KhronosGroup/SPIRV-Tools` at `199cb207b911501ddd76dcddf100a6e21c15ef23` | Apache-2.0 | SPIR-V assembly | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream SPIR-V assembly fixture unchanged. |

## Source adjustments

The DirectX, Metal performance, NVIDIA CUDA Samples, OpenCL-SDK, Rust-GPU
VulkanShaderExamples, SPIRV-Tools, raylib, SaschaWillems triangle, and Slang
cases keep upstream source files unchanged apart from repository formatting
checks. The Vulkan Samples, Apple, and Rust-GPU/rust-gpu graphics cases are
reduced source slices copied from fixture-backed upstream examples so the demo
remains small and deterministic. The reductions remove unrelated code around
the shader construct being demonstrated; they do not patch translator output.

The `glfw/glfw` OpenGL triangle shader strings were tested as a candidate and
exposed a CrossGL intermediate keyword collision: the fragment shader uses
`fragment` as a user output variable, which currently fails downstream target
generation after GLSL-to-CrossGL conversion. That translator issue is tracked
in issue #766, and the case is intentionally not checked in until keyword-safe
identifier handling preserves the shader output.

The `KhronosGroup/glslang` specialization-constant vertex shader from
`Test/spv.specConstant.vert` at `98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515`
was tested as a candidate and exposed target lowering gaps for
`layout(constant_id)`, `gl_MaxImageUnits`, and fixed-size stage-input arrays.
That translator issue is tracked in issue #780, and the case is intentionally
not checked in until generated target artifacts preserve the specialization
constant semantics without placeholder output.

The `shader-slang/slang` default-parameter compute shader from
`tests/compute/default-parameter.slang` at
`adc996670ec281aa8a4ee131f30b324648cbbe60` was tested as a candidate and
exposed target lowering gaps for default function parameters in OpenGL and
Metal output. That translator issue is tracked in issue #781, and the case is
intentionally not checked in until generated source targets preserve the
default-argument call semantics.

The `microsoft/DirectX-Graphics-Samples` HelloTexture shader from
`Samples/Desktop/D3D12HelloWorld/src/HelloTexture/shaders.hlsl` at
`31ae3c91160d8634264004cdaf4e41a99c41243e` was tested as a candidate and
exposed target lowering gaps for HLSL texture sampling and stage IO semantics
in Metal and OpenGL output. That translator issue is tracked in issue #783,
and the case is intentionally not checked in until generated source targets
compile with platform validators.

The `microsoft/DirectXShaderCompiler` scalar-splat compute test was tested as
a candidate and exposed target semantic gaps for HLSL scalar swizzles and
`groupshared` storage. That translator issue is tracked in issue #767, and the
case is intentionally not checked in until target output preserves the compute
semantics rather than emitting placeholder comments.

The `NVIDIA/cuda-samples` vector-add NVRTC kernel was retested after issue #772
closed and is now checked for Metal and Vulkan output. The upstream host
launcher is intentionally not included; rewriting launch configuration, memory
allocation, and data-transfer code is a runtime porting task outside this demo
scope.

The `Rust-GPU/VulkanShaderExamples` conservative raster triangle-overlay shader
was retested after issue #776 closed and is now checked for Metal and Vulkan
output. The `Rust-GPU/rust-gpu` graphics stage-input slice is checked for
CrossGL, OpenGL, Metal, and Vulkan output. Rust-GPU compute examples still
need Rust `Option<T>`, fixed-size array, and unsigned-constant lowering for
Metal and SPIR-V targets; that translator issue is tracked in issue #775.

The `ROCm/rocm-examples` bit-extract HIP kernel was retested after issue #778
closed. The generated artifacts now preserve a compute entry point, but Metal
still needs scalar kernel-parameter lowering and host `main` filtering, and
SPIR-V still needs host-entry filtering; that follow-up is tracked in issue
#795. HIP candidates remain intentionally excluded until target artifacts
compile directly and preserve only the relevant kernel entries.

The `modular/modular` Mojo GPU vector-add example was tested as a candidate.
It translates to CrossGL, but Metal and SPIR-V artifact generation still fails
when Mojo identifier AST nodes cross the target-generation path; that
translator issue is tracked in issue #798. The candidate is intentionally not
checked in until platform target artifacts can be generated and validated.

The `KhronosGroup/OpenCL-SDK` SAXPY kernel was retested after issue #751 closed
and is now checked for Metal and Vulkan output. OpenGL output still needs an
explicit cast when assigning `gl_GlobalInvocationID.x` to the signed index
declared by the source kernel; that follow-up is tracked in issue #768.

The `SaschaWillems/Vulkan` headless compute shader was tested as a candidate
and exposed storage-buffer lowering gaps for OpenGL, DirectX, and Metal
artifacts. That translator issue is tracked in issue #756, and the case is
intentionally not checked in until those target artifacts preserve buffer
resource access.

## Generated artifacts

Each case stores reference artifacts under `crosstl-out/`. Portability reports
are not committed because they include machine-local absolute project roots.
The demo runner and CI workflow generate fresh reports for every run and can
write them to a separate reports directory for inspection. The runner disables
optional code formatting during generation so artifact comparisons do not
depend on host `clang-format` availability.

## Third-party notices

The source slices remain under their upstream project licenses. See
`THIRD_PARTY_NOTICES.md` for repository, license, and source URL details.
