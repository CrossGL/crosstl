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
| `vulkan-samples-dynamic-line-grid` | `KhronosGroup/Vulkan-Samples` at `ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a` | Apache-2.0 | GLSL | CrossGL, Metal, DirectX, Vulkan | Uses the reduced fragment shader already covered by backend fixture provenance. OpenGL smoke validation for this case is tracked in issue #745. |
| `apple-modern-rendering-mesh-viewdir` | `donaldwuid/apple_metal_sample_code` at `0bc50e5b3670b3169855ab260e8da5ff07b53749` | MIT | Metal | CrossGL, Metal, DirectX, Vulkan | Uses a reduced shader slice that keeps the relevant vertex-stage type conversion. OpenGL output is tracked in issue #746. |
| `metal-performance-testing-matmul` | `bkvogel/metal_performance_testing` at `b467b4b1dee0f7d9d43bda13856306ca3f1baea5` | BSD-style | Metal | CrossGL, Metal, Vulkan | Uses the upstream Metal kernel and its shared parameter header. DirectX constant-parameter lowering is tracked in issue #755. |
| `opencl-sdk-saxpy` | `KhronosGroup/OpenCL-SDK` at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` | Apache-2.0 | OpenCL | CrossGL, Metal, Vulkan | Uses the upstream SAXPY compute kernel unchanged. OpenGL index-cast lowering is tracked in issue #768. |
| `raylib-base-fragment` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, Metal, Vulkan | Uses the upstream base fragment shader unchanged. OpenGL binding qualifier/version alignment is tracked in issue #765. |
| `sascha-willems-vulkan-conservative-triangle` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream conservative raster triangle vertex shader without semantic edits. DirectX semantic lowering remains outside this checked target subset. |
| `slang-hello-world-compute` | `shader-slang/slang` at `29e69b0bf626f87500be73a7fb3764db25658c66` | Apache-2.0 WITH LLVM-exception | Slang | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream compute shader unchanged. |
| `spirv-tools-basic-src` | `KhronosGroup/SPIRV-Tools` at `199cb207b911501ddd76dcddf100a6e21c15ef23` | Apache-2.0 | SPIR-V assembly | CrossGL, Metal, Vulkan | Uses the upstream SPIR-V assembly fixture unchanged. OpenGL builtin interface lowering is not included in this checked target subset. |

## Source adjustments

The DirectX, Metal performance, OpenCL-SDK, SPIRV-Tools, raylib,
SaschaWillems triangle, and Slang cases keep upstream source files unchanged
apart from repository formatting checks. The Vulkan Samples and Apple cases are
reduced source slices copied from fixture-backed upstream examples so the demo
remains small and deterministic. The reductions remove unrelated code around
the shader construct being demonstrated; they do not patch translator output.

The complete `lonelydevil/vulkan-tutorial-C-implementation` shader pair was
retested after project artifact naming was fixed in issue #743. The pair now
emits distinct artifacts for `shader.vert` and `shader.frag`, but the vertex
shader uses Vulkan-style `gl_VertexIndex`; OpenGL and DirectX target lowering
for that builtin is tracked in issue #763, and the case remains intentionally
unchecked until target output preserves the vertex-index semantics.

The `glfw/glfw` OpenGL triangle shader strings were tested as a candidate and
exposed a CrossGL intermediate keyword collision: the fragment shader uses
`fragment` as a user output variable, which currently fails downstream target
generation after GLSL-to-CrossGL conversion. That translator issue is tracked
in issue #766, and the case is intentionally not checked in until keyword-safe
identifier handling preserves the shader output.

The `microsoft/DirectXShaderCompiler` scalar-splat compute test was tested as
a candidate and exposed target semantic gaps for HLSL scalar swizzles and
`groupshared` storage. That translator issue is tracked in issue #767, and the
case is intentionally not checked in until target output preserves the compute
semantics rather than emitting placeholder comments.

The `NVIDIA/cuda-samples` vector-add CUDA kernel was tested as a reduced source
slice from `cpp/0_Introduction/vectorAdd/vectorAdd.cu` at
`b7c5481c556c3fe98db060207ecaa41a4b9a9abc`. Metal and SPIR-V target output
still needs CUDA compute builtin lowering and valid target entry-point
emission; that translator issue is tracked in issue #772, and the case is
intentionally not checked in until target artifacts validate directly.

Rust-GPU graphics and compute examples were tested from `Rust-GPU/rust-gpu`
and `Rust-GPU/VulkanShaderExamples`. The graphics example needs target-native
entry-point IO lowering for vertex index, position, and fragment color outputs;
that translator issue is tracked in issue #776. The compute examples need Rust
`Option<T>`, fixed-size array, and unsigned-constant lowering for Metal and
SPIR-V targets; that translator issue is tracked in issue #775. These cases are
intentionally not checked in until generated target artifacts preserve the
shader semantics and validate directly.

The `ROCm/rocm-examples` bit-extract HIP kernel and the `ROCm/hip-tests` Set
kernel were tested as candidates. Metal and SPIR-V output still needs HIP
compute entry-point, storage-buffer, and invocation-index lowering; that
translator issue is tracked in issue #778, and the cases are intentionally not
checked in until target artifacts preserve compute semantics.

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
