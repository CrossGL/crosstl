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
| `directx-graphics-samples-hello-const-buffers` | `microsoft/DirectX-Graphics-Samples` at `31ae3c91160d8634264004cdaf4e41a99c41243e` | MIT | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Hello ConstBuffers shader file unchanged. Host constant-buffer setup remains outside the demo scope. |
| `directx-graphics-samples-hello-texture` | `microsoft/DirectX-Graphics-Samples` at `31ae3c91160d8634264004cdaf4e41a99c41243e` | MIT | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Hello Texture shader file without source edits. Host texture setup remains outside the demo scope. |
| `directx-shader-compiler-groupshared-splat` | `microsoft/DirectXShaderCompiler` at `517dd5eb5d8cbb46c15fc1230acac1d2f4779092` | University of Illinois/NCSA | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream groupshared scalar-splat compute shader unchanged. |
| `directx-shader-compiler-neg1` | `microsoft/DirectXShaderCompiler` at `d6e0ca4a0c25b13ed676c8ba16839c3eb9fcc652` | University of Illinois/NCSA | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream negated swizzle pixel shader unchanged. |
| `directx-sdk-samples-tutorial02` | `walbourn/directx-sdk-samples-reworked` at `1ad8f0f6a3e4d9be7e54ca52640ac12b6565ab0c` | MIT | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Direct3D 11 Tutorial02 effect include unchanged. |
| `diligent-samples-tutorial02-cube` | `DiligentGraphics/DiligentSamples` at `30b94f26e7d10cde0be48c75a2c252185f564b69` | Apache-2.0 | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Tutorial02 cube vertex and pixel shader pair with repository whitespace normalization. |
| `diligent-samples-vrs-cube-vertex` | `DiligentGraphics/DiligentSamples` at `30b94f26e7d10cde0be48c75a2c252185f564b69` | Apache-2.0 | GLSL | CrossGL, OpenGL, Vulkan | Uses the upstream VRS cube vertex shader unchanged. The paired fragment-density stage remains outside the checked set because the translator now diagnoses it as unsupported. |
| `glslang-push-constant-vertex` | `KhronosGroup/glslang` at `98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515` | BSD-style | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream push-constant vertex shader unchanged. |
| `glslang-spec-constant-vertex` | `KhronosGroup/glslang` at `98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515` | BSD-style | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream specialization-constant vertex shader unchanged. Source-target output records fallback literals where native specialization IDs cannot be preserved. |
| `godot-betsy-alpha-stitch` | `godotengine/godot` at `3df26a02c446710c979daa541b74f87edeca81b0` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Removes the Godot shader-section marker so the compute shader is standalone GLSL. |
| `libgdx-batch-shader` | `libgdx/libgdx` at `846d63a746e4604a7699133f803ff844fdc8c9fe` | Apache-2.0 | GLSL ES | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream batch shader pair unchanged apart from line-ending and trailing-whitespace normalization. |
| `lonelydevil-vulkan-tutorial-triangle` | `lonelydevil/vulkan-tutorial-C-implementation` at `780ff146a6eccd7064a10e86363f3c2f7323825d` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream triangle shader pair unchanged. |
| `monogame-sprite-effect` | `MonoGame/MonoGame` at `d4893ac09e06bc203792d01d6f151f1891cc1ab5` | MS-PL and MIT | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream SpriteEffect source and macro include with whitespace normalization. |
| `spirv-cross-round-fragment` | `KhronosGroup/SPIRV-Cross` at `146679ff8255a6068518685599d7fb8761d1b570` | Apache-2.0 | GLSL | CrossGL, OpenGL, Vulkan | Uses the upstream fragment reference shader unchanged. |
| `vulkan-samples-dynamic-line-grid` | `KhronosGroup/Vulkan-Samples` at `ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a` | Apache-2.0 | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the reduced fragment shader already covered by backend fixture provenance. |
| `angle-simple-texture-2d` | `google/angle` at `52232eaf409a28d77947df5622af274e1ef770c6` | BSD-style | GLSL ES | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses extracted upstream SimpleTexture2D shader strings. |
| `apple-modern-rendering-mesh-viewdir` | `donaldwuid/apple_metal_sample_code` at `0bc50e5b3670b3169855ab260e8da5ff07b53749` | MIT | Metal | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses a reduced shader slice that keeps the relevant vertex-stage type conversion. |
| `arm-opengl-es-sdk-cube` | `ARM-software/opengl-es-sdk-for-android` at `c3caf759bb2e71fa9a118b3e3abd996cf00e660a` | MIT | GLSL ES | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream cube shader pair unchanged. |
| `metal-performance-testing-matmul` | `bkvogel/metal_performance_testing` at `b467b4b1dee0f7d9d43bda13856306ca3f1baea5` | BSD-style | Metal | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Metal kernel and its shared parameter header. |
| `nvidia-cuda-samples-vector-add` | `NVIDIA/cuda-samples` at `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` | BSD-style with CUDA EULA reference | CUDA | CrossGL, Metal, DirectX, Vulkan | Uses the upstream NVRTC vectorAdd kernel unchanged. Host launch and memory-management integration remain outside the demo scope. |
| `nvpro-vk-mini-samples-rectangle` | `nvpro-samples/vk_mini_samples` at `994ac9f446ef44962c563b9600c8e9f117a3725d` | Apache-2.0 | GLSL | CrossGL, Metal, OpenGL, Vulkan | Uses the upstream rectangle shader pair unchanged. |
| `ogl-samples-flat-color` | `g-truc/ogl-samples` at `38cada7a9458864265e25415ae61586d500ff5fc` | MIT | GLSL | CrossGL, Metal, OpenGL, Vulkan | Uses the upstream GLSL 330 flat-color shader pair unchanged. |
| `openframeworks-noise-shader` | `openframeworks/openFrameworks` at `63eb03828c40de713b85db7810f1c519d8b9b0cc` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream noise shader pair with whitespace normalization. |
| `opencl-sdk-saxpy` | `KhronosGroup/OpenCL-SDK` at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` | Apache-2.0 | OpenCL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream SAXPY compute kernel unchanged. |
| `rocm-examples-add-kernel` | `ROCm/rocm-examples` at `cf369da68f209c315074204bd0eb61d1a5c015d1` | MIT | HIP | CrossGL, Metal, Vulkan | Uses the upstream sphinx-marked add-kernel slice. Host HIP runtime setup remains outside the demo scope. |
| `raylib-base-fragment` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream base fragment shader unchanged. |
| `raylib-base-vertex` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream base vertex shader unchanged. |
| `raylib-lighting-shader-pair` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream lighting vertex and fragment shaders unchanged. |
| `renderdoc-vktext-fragment` | `baldurk/renderdoc` at `6660344c3d8024dc5107afa2115c5035ceb85533` | MIT | GLSL | CrossGL, OpenGL, Vulkan | Uses the upstream Vulkan text fragment shader unchanged. |
| `rust-gpu-compute-collatz` | `Rust-GPU/rust-gpu` at `36e3348cdc2f824afec64b3b5af5d369d98a4c0d` | Apache-2.0 OR MIT | Rust-GPU | CrossGL, Metal, Vulkan | Uses the upstream compute shader unchanged. |
| `rust-gpu-graphics-stage-inputs` | `Rust-GPU/rust-gpu` at `36e3348cdc2f824afec64b3b5af5d369d98a4c0d` | Apache-2.0 OR MIT | Rust-GPU | CrossGL, OpenGL, Metal, Vulkan | Uses a reduced graphics shader slice that keeps the plain vertex input and fragment color path. |
| `rust-gpu-vulkan-examples-triangle-overlay` | `Rust-GPU/VulkanShaderExamples` at `b29a37eb46802b5ea6882af4808d6887fc184581` | MIT | Rust-GPU | CrossGL, Metal, Vulkan | Uses the upstream conservative raster triangle-overlay shader unchanged. |
| `sascha-willems-vulkan-conservative-triangle` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream conservative raster triangle vertex shader without semantic edits. DirectX `SV_Position` lowering is tracked in issue #1196. |
| `sascha-willems-vulkan-headless-compute` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream headless compute shader unchanged. |
| `slang-hello-world-compute` | `shader-slang/slang` at `29e69b0bf626f87500be73a7fb3764db25658c66` | Apache-2.0 WITH LLVM-exception | Slang | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream compute shader unchanged. |
| `spirv-tools-basic-src` | `KhronosGroup/SPIRV-Tools` at `199cb207b911501ddd76dcddf100a6e21c15ef23` | Apache-2.0 | SPIR-V assembly | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream SPIR-V assembly fixture unchanged. |
| `vulkan-tools-cube` | `KhronosGroup/Vulkan-Tools` at `68749eafbf27114a1dd807d6c870e53306673e64` | Apache-2.0 | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream cube demo shader pair unchanged. |

## Source adjustments

The ARM OpenGL ES SDK, DiligentSamples Tutorial02 cube, DiligentSamples VRS
vertex, DirectX, DirectX SDK Samples, DirectXShaderCompiler, glslang, Metal
performance, NVIDIA CUDA Samples, MonoGame, nvpro-samples, ogl-samples,
OpenCL-SDK, openFrameworks, RenderDoc, Rust-GPU VulkanShaderExamples,
SPIRV-Cross, SPIRV-Tools, Vulkan-Tools, raylib, SaschaWillems triangle, and
Slang cases keep upstream source files unchanged apart from repository
formatting checks. The DirectX Hello Texture shader was retested after issue
#783 closed and is now checked for OpenGL, Metal, DirectX, and Vulkan output.
The DirectX Hello ConstBuffers shader is checked for the same target set while
leaving host-side constant-buffer allocation outside this source-focused demo.
The SaschaWillems headless compute shader was retested after issue #756 and
issue #780 closed and is now checked for OpenGL, Metal, DirectX, and Vulkan
output. The Vulkan Samples, Apple, ROCm add-kernel, and Rust-GPU/rust-gpu
graphics cases are reduced source slices copied from fixture-backed upstream
examples so the demo remains small and deterministic. The reductions remove
unrelated code around the shader construct being demonstrated; they do not
patch translator output.

The `google/angle` SimpleTexture2D case extracts the upstream GLSL ES shader
strings from `samples/simple_texture_2d/SimpleTexture2D.cpp` into standalone
shader files so the project translation runner can treat them as ordinary
translation units. The extraction adds only license/provenance comments and
does not change shader semantics. OpenGL output was retested after issue #820
closed and is now checked with the other source targets.

The `libgdx/libgdx` batch shader pair keeps the upstream shader statements
unchanged, with line endings and trailing whitespace normalized by repository
formatting. OpenGL output was retested after issue #820 closed and is now
checked with the other source targets.

The `MonoGame/MonoGame` SpriteEffect case keeps the upstream effect source and
macro include semantically unchanged, with indentation normalized by repository
formatting. DirectX output is checked after issue #869 restored valid HLSL
resource declarations. Metal output was retested after issue #873 closed and is
now checked, warning cleanup was retested after issue #917 closed, and OpenGL
output was retested after issue #947 closed.

The `openframeworks/openFrameworks` noise shader pair keeps the upstream shader
statements unchanged, with indentation and trailing whitespace normalized by
repository formatting. OpenGL and Metal output were retested after issue #840
and issue #841 closed and are now checked. DirectX output was retested after
issue #875 closed and is now checked.

The `godotengine/godot` Betsy alpha-stitch case removes the upstream
`#[compute]` shader-section marker before `#version`. That marker is part of
Godot's shader packaging format and is not valid standalone GLSL; shader
statements, resource declarations, and entry-point logic are unchanged. Vulkan
and DirectX output were retested after issue #829 closed and are now checked.
Metal output was retested after issue #887 closed and is now checked.

The `glfw/glfw` OpenGL triangle shader strings were tested as a candidate and
originally exposed the CrossGL intermediate keyword collision tracked in issue
#766. Retesting after issue #1159 closed shows the generated Metal fragment
shader now escapes the `fragment` stage keyword collision and compiles
directly. The case remains a future expansion candidate because it is not yet
stored with the checked demo corpus and third-party notices.

The checked `KhronosGroup/glslang` push-constant vertex shader was retested
after issue #813 and issue #856 closed and is now checked for DirectX and
Metal output alongside OpenGL and Vulkan. The specialization-constant vertex
shader was retested after issue #780 closed and is now checked for CrossGL,
OpenGL, Metal, and Vulkan output. Generated source targets use the source
defaults for specialization constants when native specialization IDs cannot be
represented directly. DirectX output was retested after issue #1154 closed and
is now checked with the other generated targets.

The `KhronosGroup/Vulkan-Samples` dynamic line grid fragment shader was
retested after issue #922 closed and is now checked for CrossGL, OpenGL,
Metal, and Vulkan. DirectX output was retested after issue #959 closed and is
now checked with the other generated targets.

The `donaldwuid/apple_metal_sample_code` mesh view-direction slice is checked
through CrossGL, OpenGL, Metal, DirectX, and Vulkan. Returned `viewDir`
preservation was retested after issue #951 closed, OpenGL and Vulkan output
were restored after issues #971 and #972 closed, and Metal resource attributes
were retested after issue #988 closed.

The `DiligentGraphics/DiligentSamples` Tutorial02 Cube shaders keep the
upstream shader statements and entry points unchanged, with trailing whitespace
normalized by repository formatting. The project config maps the nonstandard
`.vsh` and `.psh` file extensions to the DirectX backend, and the case is
checked for CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue
#1147 closed.

The checked `DiligentGraphics/DiligentSamples` VRS cube vertex shader is
included for CrossGL, OpenGL, and Vulkan output. The paired fragment shader
uses `GL_EXT_fragment_invocation_density` and `gl_FragSizeEXT`; generated
Metal and HLSL previously kept that GLSL built-in as an undeclared identifier,
and generated SPIR-V lost the built-in value. Retesting after issue #826 closed
shows the fragment stage now fails with structured unsupported-feature
diagnostics, so it remains excluded until fragment-density semantics are
lowered for target backends.

The `g-truc/ogl-samples` flat-color shader pair is checked for CrossGL,
Metal, OpenGL, and Vulkan output.

The `nvpro-samples/vk_mini_samples` rectangle shader pair is checked for
CrossGL, Metal, OpenGL, and Vulkan output.

The `KhronosGroup/Vulkan-Tools` cube demo shaders are checked for CrossGL,
OpenGL, Metal, DirectX, and Vulkan output after issue #819 restored HLSL
reserved-keyword handling and issue #975 restored distinct DirectX vertex
output semantics.

The `ARM-software/opengl-es-sdk-for-android` cube shaders are checked for
CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue #820 restored
legacy `gl_FragColor` lowering.

The `KhronosGroup/OpenCL-SDK` reduce kernel from `samples/core/reduce/reduce.cl`
at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` was tested as a candidate and
retested after issue #811 closed. The project pipeline now reports structured
unsupported-lowering diagnostics for unresolved reduction helpers, local
pointer helper parameters, and event/local-memory builtins instead of checking
in invalid target artifacts. The case remains excluded until those OpenCL
semantics are lowered or the source is reduced to a supported kernel shape.

The `microsoft/DirectX-Graphics-Samples` HelloConstBuffers shader from
`Samples/Desktop/D3D12HelloWorld/src/HelloConstBuffers/shaders.hlsl` at
`31ae3c91160d8634264004cdaf4e41a99c41243e` was tested as a candidate. It was
retested after issue #812 closed and now emits directly compilable Metal
attribute bindings for the candidate shader. It remains documented here as a
candidate for a future demo expansion rather than a current checked case.

The `shader-slang/slang` default-parameter compute shader from
`tests/compute/default-parameter.slang` at
`adc996670ec281aa8a4ee131f30b324648cbbe60` was tested as a candidate and
exposed target lowering gaps for default function parameters in OpenGL and
Metal output. Retesting after issue #781 closed shows OpenGL, Metal, and
SPIR-V output now validate, so the source is a candidate for a future demo
expansion.

The `microsoft/DirectXShaderCompiler` scalar-splat compute test keeps the
upstream source unchanged and is checked for CrossGL, OpenGL, Metal, DirectX,
and Vulkan output. Retesting after issue #767 closed shows scalar splats are
preserved in SPIR-V, and retesting after issue #1149 closed shows generated
Metal now lowers the program-scope `groupshared` value to kernel-local
`threadgroup` storage and validates directly.

The `NVIDIA/cuda-samples` vector-add NVRTC kernel was retested after issue #772
closed and is now checked for Metal and Vulkan output. DirectX output was
retested after issue #1183 closed and is now checked as a compute shader. The
upstream host launcher is intentionally not included; rewriting launch
configuration, memory allocation, and data-transfer code is a runtime porting
task outside this demo scope.

The `Rust-GPU/VulkanShaderExamples` conservative raster triangle-overlay shader
was retested after issue #776 closed and is now checked for Metal and Vulkan
output. The `Rust-GPU/rust-gpu` compute Collatz shader was retested after
issue #809 closed and is now checked for Metal and Vulkan output. The
`Rust-GPU/rust-gpu` graphics stage-input slice is checked for CrossGL, OpenGL,
Metal, and Vulkan output. Full Rust-GPU crate builds, host-side dispatch, and
runtime validation remain outside this source-focused demo scope.

The `ROCm/rocm-examples` add-kernel case uses only the upstream
`[sphinx-kernel-start]` to `[sphinx-kernel-end]` source section. The full sample
host `main()`, HIP runtime calls, and launch configuration are runtime
integration work outside this shader translation demo. The earlier
`ROCm/rocm-examples` bit-extract HIP kernel was retested after issue #795
closed; generated Metal now compiles and generated SPIR-V contains only the
translated compute entry point. It remains a candidate for a future demo
expansion.

The `modular/modular` Mojo GPU vector-add example was tested as a candidate.
It now generates Metal and SPIR-V artifacts after issue #798 closed, but those
artifacts previously contained unresolved Mojo host/runtime constructs and
failed direct target validation. Retesting after issue #1148 closed shows the
project pipeline now rejects that unresolved host/runtime surface with
structured diagnostics. The candidate is intentionally not checked in until a
shader-only kernel slice can be translated and validated as platform target
source.

The `KhronosGroup/OpenCL-SDK` SAXPY kernel was retested after issue #751 and
issue #768 closed and is now checked for OpenGL, Metal, and Vulkan output.

The `bkvogel/metal_performance_testing` matmul kernel is checked for CrossGL,
OpenGL, Metal, DirectX, and Vulkan output after issue #1158 restored OpenGL
buffer resource declarations and issue #1191 restored DXC-valid
`RWStructuredBuffer` writes.

The `SaschaWillems/Vulkan` headless compute shader was retested after issue
#780 closed and is now checked for OpenGL, Metal, DirectX, and Vulkan output.

## Generated artifacts

Each case stores reference artifacts under `crosstl-out/`. Portability reports
are not committed because they include machine-local absolute project roots.
The demo runner and CI workflow generate fresh reports for every run and can
write them to a separate reports directory for inspection. The runner disables
optional code formatting during generation so artifact comparisons do not
depend on host `clang-format` availability.

## Third-party notices

The source slices remain under their upstream project licenses. See
`THIRD_PARTY_NOTICES.md` for repository, license, and source path details.
Exact pinned source URLs are recorded in each case's `corpus.json` manifest.
