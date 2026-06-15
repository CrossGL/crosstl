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
| `diligent-samples-vrs-cube` | `DiligentGraphics/DiligentSamples` at `30b94f26e7d10cde0be48c75a2c252185f564b69` | Apache-2.0 | GLSL | CrossGL, OpenGL, DirectX, Vulkan | Uses the upstream VRS cube vertex and fragment-density shader pair unchanged. Metal has no fragment-size input equivalent. |
| `glfw-opengl-triangle` | `glfw/glfw` at `567b1ec2442d59525e24c19e8d413df6baf02496` | Zlib | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream OpenGL triangle shader strings extracted from `triangle-opengl.c`. |
| `glslang-push-constant-vertex` | `KhronosGroup/glslang` at `98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515` | BSD-style | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream push-constant vertex shader unchanged. |
| `glslang-spec-constant-vertex` | `KhronosGroup/glslang` at `98beacdbe5d99f4ac5e4c58bc02bb16c6aeee515` | BSD-style | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream specialization-constant vertex shader unchanged. Source-target output records fallback literals where native specialization IDs cannot be preserved. |
| `godot-betsy-alpha-stitch` | `godotengine/godot` at `3df26a02c446710c979daa541b74f87edeca81b0` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Removes the Godot shader-section marker so the compute shader is standalone GLSL. |
| `libgdx-batch-shader` | `libgdx/libgdx` at `846d63a746e4604a7699133f803ff844fdc8c9fe` | Apache-2.0 | GLSL ES | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream batch shader pair unchanged apart from line-ending and trailing-whitespace normalization. |
| `lonelydevil-vulkan-tutorial-triangle` | `lonelydevil/vulkan-tutorial-C-implementation` at `780ff146a6eccd7064a10e86363f3c2f7323825d` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream triangle shader pair unchanged. |
| `monogame-sprite-effect` | `MonoGame/MonoGame` at `d4893ac09e06bc203792d01d6f151f1891cc1ab5` | MS-PL and MIT | DirectX/HLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream SpriteEffect source and macro include with whitespace normalization. |
| `spirv-cross-round-fragment` | `KhronosGroup/SPIRV-Cross` at `146679ff8255a6068518685599d7fb8761d1b570` | Apache-2.0 | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream fragment reference shader unchanged. |
| `vulkan-samples-dynamic-line-grid` | `KhronosGroup/Vulkan-Samples` at `ab1e93d4a5dadf4c804fb6abbbe0b27dfa912b5a` | Apache-2.0 | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the reduced fragment shader already covered by backend fixture provenance. |
| `angle-simple-texture-2d` | `google/angle` at `52232eaf409a28d77947df5622af274e1ef770c6` | BSD-style | GLSL ES | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses extracted upstream SimpleTexture2D shader strings. |
| `apple-modern-rendering-mesh-viewdir` | `donaldwuid/apple_metal_sample_code` at `0bc50e5b3670b3169855ab260e8da5ff07b53749` | MIT | Metal | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses a reduced shader slice that keeps the relevant vertex-stage type conversion. |
| `arm-opengl-es-sdk-cube` | `ARM-software/opengl-es-sdk-for-android` at `c3caf759bb2e71fa9a118b3e3abd996cf00e660a` | MIT | GLSL ES | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream cube shader pair unchanged. |
| `metal-performance-testing-matmul` | `bkvogel/metal_performance_testing` at `b467b4b1dee0f7d9d43bda13856306ca3f1baea5` | BSD-style | Metal | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Metal kernel and its shared parameter header. |
| `nvidia-cuda-samples-vector-add` | `NVIDIA/cuda-samples` at `b7c5481c556c3fe98db060207ecaa41a4b9a9abc` | BSD-style with CUDA EULA reference | CUDA | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream NVRTC vectorAdd kernel unchanged. Host launch and memory-management integration remain outside the demo scope. |
| `nvpro-vk-mini-samples-rectangle` | `nvpro-samples/vk_mini_samples` at `994ac9f446ef44962c563b9600c8e9f117a3725d` | Apache-2.0 | GLSL | CrossGL, Metal, OpenGL, DirectX, Vulkan | Uses the upstream rectangle shader pair unchanged. |
| `ogl-samples-flat-color` | `g-truc/ogl-samples` at `38cada7a9458864265e25415ae61586d500ff5fc` | MIT | GLSL | CrossGL, Metal, OpenGL, DirectX, Vulkan | Uses the upstream GLSL 330 flat-color shader pair unchanged. |
| `openframeworks-noise-shader` | `openframeworks/openFrameworks` at `63eb03828c40de713b85db7810f1c519d8b9b0cc` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream noise shader pair with whitespace normalization. |
| `opencl-sdk-reduce` | `KhronosGroup/OpenCL-SDK` at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` | Apache-2.0 | OpenCL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream reduce compute kernel unchanged. |
| `opencl-sdk-saxpy` | `KhronosGroup/OpenCL-SDK` at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` | Apache-2.0 | OpenCL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream SAXPY compute kernel unchanged. |
| `rocm-examples-add-kernel` | `ROCm/rocm-examples` at `cf369da68f209c315074204bd0eb61d1a5c015d1` | MIT | HIP | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream sphinx-marked add-kernel slice. Host HIP runtime setup remains outside the demo scope. |
| `rocm-examples-bit-extract` | `ROCm/rocm-examples` at `cf369da68f209c315074204bd0eb61d1a5c015d1` | MIT | HIP | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream bit-extract kernel slice with the AMD HIP branch selected through project defines. Host HIP runtime setup remains outside the demo scope. |
| `raylib-base-fragment` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream base fragment shader unchanged. |
| `raylib-base-vertex` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream base vertex shader unchanged. |
| `raylib-lighting-shader-pair` | `raysan5/raylib` at `94897c4eca842673bad16ab03ad776a0a2255b14` | zlib/libpng | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream lighting vertex and fragment shaders unchanged. |
| `renderdoc-vktext-fragment` | `baldurk/renderdoc` at `6660344c3d8024dc5107afa2115c5035ceb85533` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream Vulkan text fragment shader unchanged. |
| `rust-gpu-compute-collatz` | `Rust-GPU/rust-gpu` at `36e3348cdc2f824afec64b3b5af5d369d98a4c0d` | Apache-2.0 OR MIT | Rust-GPU | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream compute shader unchanged. |
| `rust-gpu-graphics-stage-inputs` | `Rust-GPU/rust-gpu` at `36e3348cdc2f824afec64b3b5af5d369d98a4c0d` | Apache-2.0 OR MIT | Rust-GPU | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses a reduced graphics shader slice that keeps the plain vertex input and fragment color path. |
| `rust-gpu-vulkan-examples-triangle-overlay` | `Rust-GPU/VulkanShaderExamples` at `b29a37eb46802b5ea6882af4808d6887fc184581` | MIT | Rust-GPU | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream conservative raster triangle-overlay shader unchanged. |
| `sascha-willems-vulkan-conservative-triangle` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream conservative raster triangle vertex shader without semantic edits. Host conservative-rasterizer pipeline state remains outside the demo scope. |
| `sascha-willems-vulkan-headless-compute` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream headless compute shader unchanged. |
| `slang-hello-world-compute` | `shader-slang/slang` at `29e69b0bf626f87500be73a7fb3764db25658c66` | Apache-2.0 WITH LLVM-exception | Slang | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream compute shader unchanged. |
| `slang-default-parameter-compute` | `shader-slang/slang` at `adc996670ec281aa8a4ee131f30b324648cbbe60` | Apache-2.0 WITH LLVM-exception | Slang | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream default-parameter compute shader with repository newline normalization. |
| `spirv-tools-basic-src` | `KhronosGroup/SPIRV-Tools` at `199cb207b911501ddd76dcddf100a6e21c15ef23` | Apache-2.0 | SPIR-V assembly | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream SPIR-V assembly fixture unchanged. |
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

The ROCm add-kernel DirectX artifact is checked after HIP pointer-parameter
lowering landed in #1219. The SPIR-V Tools basic-source DirectX artifact is
checked after the corresponding DirectX lowering fix landed on main. OpenCL
SAXPY DirectX output is checked after issue #1227 closed, and Rust-GPU compute
CGL/Vulkan/DirectX output is checked after issue #1232 and issue #1230 closed.

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
directly. The extracted shader strings are now checked for CrossGL, OpenGL,
Metal, DirectX, and Vulkan output after issue #1304 restored GLSL 110
interface-qualifier compatibility.

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

The `DiligentGraphics/DiligentSamples` VRS cube vertex and fragment-density
shaders are checked for CrossGL, OpenGL, and Vulkan output after issue #1190
added Vulkan `FragSizeEXT` modeling for `GL_EXT_fragment_invocation_density`.
DirectX output lowers the fragment-size path through `SV_ShadingRate` and is
checked with a VRS-capable DXC pixel profile after issue #1235 closed. Metal
remains excluded because the target does not expose a fragment-size input
equivalent for this shader. The demo keeps host VRS render-pass attachment and
pipeline-state integration out of scope.

The `g-truc/ogl-samples` flat-color shader pair is checked for CrossGL,
Metal, OpenGL, and Vulkan output.

The `nvpro-samples/vk_mini_samples` rectangle shader pair is checked for
CrossGL, Metal, OpenGL, DirectX, and Vulkan output.

The `KhronosGroup/SPIRV-Cross` round fragment shader is checked for CrossGL,
OpenGL, Metal, DirectX, and Vulkan output.

The `KhronosGroup/Vulkan-Tools` cube demo shaders are checked for CrossGL,
OpenGL, Metal, and DirectX output after issue #819 restored HLSL
reserved-keyword handling and issue #975 restored distinct DirectX vertex
output semantics. Vulkan output is checked after issue #1285 restored SPIR-V
overload argument typing for the fragment shader.

The `lonelydevil/vulkan-tutorial-C-implementation` triangle shader pair is
checked for CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue
#1246 restored integer access-chain index lowering.

The `ARM-software/opengl-es-sdk-for-android` cube shaders are checked for
CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue #820 restored
legacy `gl_FragColor` lowering.

The `raysan5/raylib` base shader cases and lighting shader pair are checked for
CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue #1271 restored
DirectX vector truncation for matrix-vector results. The demo still leaves
raylib's host-side uniform, texture, and material binding setup outside the
translated artifact set.

The `baldurk/renderdoc` Vulkan text fragment shader is checked for CrossGL,
OpenGL, Metal, DirectX, and Vulkan output. RenderDoc's UI text atlas setup and
render-pass integration remain outside this source-focused demo.

The `KhronosGroup/OpenCL-SDK` reduce kernel from `samples/core/reduce/reduce.cl`
at `e26922bdf54eaa9fcc31fe1f91d21b8d2bd6970f` was tested as a candidate and
is checked for CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issues
#1241, #1245, and #1251 closed. The translated case follows the default
non-subgroup reduction path; host-side work-group sizing and dispatch remain
outside this source-focused demo scope.

The `microsoft/DirectX-Graphics-Samples` HelloConstBuffers shader from
`Samples/Desktop/D3D12HelloWorld/src/HelloConstBuffers/shaders.hlsl` at
`31ae3c91160d8634264004cdaf4e41a99c41243e` was tested as a candidate. It was
retested after issue #812 closed and now emits directly compilable Metal
attribute bindings. The shader is checked for CrossGL, OpenGL, Metal, DirectX,
and Vulkan output.

The `shader-slang/slang` default-parameter compute shader from
`tests/compute/default-parameter.slang` at
`adc996670ec281aa8a4ee131f30b324648cbbe60` was tested as a candidate and
exposed target lowering gaps for default function parameters in OpenGL and
Metal output. Retesting after issue #781 closed shows OpenGL, Metal, and
SPIR-V output now validate. The source is checked for CrossGL, OpenGL, Metal,
DirectX, and Vulkan output.

The `microsoft/DirectXShaderCompiler` scalar-splat compute test keeps the
upstream source unchanged and is checked for CrossGL, OpenGL, Metal, DirectX,
and Vulkan output. Retesting after issue #767 closed shows scalar splats are
preserved in SPIR-V, and retesting after issue #1149 closed shows generated
Metal now lowers the program-scope `groupshared` value to kernel-local
`threadgroup` storage and validates directly.

The `NVIDIA/cuda-samples` vector-add NVRTC kernel was retested after issue #772
closed and is now checked for Metal and Vulkan output. DirectX output was
retested after issue #1183 closed and is now checked as a compute shader.
OpenGL output is checked after issue #1290 restored cross-platform uniform
member naming and source-map determinism for this CUDA kernel. The upstream
host launcher is intentionally not included; rewriting launch configuration,
memory allocation, and data-transfer code is a runtime porting task outside
this demo scope.

The `Rust-GPU/VulkanShaderExamples` conservative raster triangle-overlay shader
was retested after issue #776 closed and is now checked for OpenGL, Metal,
DirectX, and Vulkan output. The `Rust-GPU/rust-gpu` compute Collatz shader was
retested after issue #809, issue #1232, and issue #1230 closed and is now
checked for CrossGL, Metal, DirectX, and Vulkan output. The `Rust-GPU/rust-gpu`
compute Collatz shader is also checked for OpenGL output after issue #1071 and
issue #1221 closed. The `Rust-GPU/rust-gpu` graphics stage-input slice is
checked for CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue
#1278 closed. Full Rust-GPU crate builds, host-side dispatch, and runtime
validation remain outside this source-focused demo scope.

The `ROCm/rocm-examples` add-kernel case uses only the upstream
`[sphinx-kernel-start]` to `[sphinx-kernel-end]` source section and is checked
for CrossGL, OpenGL, Metal, DirectX, and Vulkan output after issue #1216 closed.
The bit-extract case uses the upstream kernel slice and selects the AMD HIP
branch with a project define after issue #795 closed. Both HIP sources are
checked for CrossGL, OpenGL, Metal, DirectX, and Vulkan output. Full sample
host `main()` functions, HIP runtime calls, and launch configuration remain
runtime integration work outside this shader translation demo.

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
DirectX output was retested after issue #1227 closed and is now checked as a
compute shader.

The `bkvogel/metal_performance_testing` matmul kernel is checked for CrossGL,
OpenGL, Metal, and DirectX output after issue #1158 restored OpenGL buffer
resource declarations and issue #1191 restored DXC-valid `RWStructuredBuffer`
writes. Vulkan output is checked after issue #1293 restored generated SPIR-V
store operand typing.

The `SaschaWillems/Vulkan` headless compute shader was retested after issue
#780 closed and is now checked for OpenGL, Metal, DirectX, and Vulkan output.

## Corpus manifests

Each case directory contains a `corpus.json` manifest describing the pinned
upstream sources. Every entry records a `role`:

- `translation_unit` (the default when `role` is omitted) marks a file that the
  case `crosstl.toml` translates, so it produces per-target artifacts under
  `crosstl-out/`.
- `support_file` marks a header or include that only provides context for a
  translation unit, such as `ShaderParams.h` for
  `metal-performance-testing-matmul` and `Macros.fxh` for
  `monogame-sprite-effect`. Support files are intentionally not translated and
  do not imply per-target output.

`run_demo.py` validates every manifest before translating a case:
`translation_unit` entries must match a `crosstl.toml` translation unit and
`support_file` entries must not. This keeps the recorded corpus coverage aligned
with the artifacts the demo actually generates and validates.

Cases that are reduced or otherwise adjusted from their upstream form record the
change next to the manifest with structured `adjustments` (each carrying a
`kind` and a `summary`) and an optional `outOfScope` list. For example,
`godot-betsy-alpha-stitch` records the removed Godot shader-section marker, and
the `rocm-examples-*` cases record the kernel slicing and the
`__HIP_PLATFORM_AMD__` define selection. The runner validates that any declared
adjustments are non-empty so the rationale stays visible alongside the fixture.

### Source-map offsets

Artifact comparison preserves source-map offset fields (`offset`, `endOffset`,
and `length`) in `*.source-remap.json`, so generated offset drift fails
verification instead of passing silently. Demo sources and artifacts are pinned
to LF through `demos/open-source-porting/.gitattributes` so these byte offsets
stay identical across Linux, macOS, and Windows.

## Generated artifacts

Each case stores reference artifacts under `crosstl-out/`. Portability reports
are not committed because they include machine-local absolute project roots.
The demo runner and CI workflow generate fresh reports for every run and can
write them to a separate reports directory for inspection. The runner disables
optional code formatting during generation so artifact comparisons do not
depend on host `clang-format` availability.

### Inspecting generated reports

Use `--reports-dir` when a demo check fails or when you want the generated
reports next to local logs:

```bash
python demos/open-source-porting/run_demo.py \
  --check \
  --case directx-shader-compiler-neg1 \
  --target cgl \
  --reports-dir /tmp/crosstl-demo-reports
```

During a `--check` run, each case is copied into a temporary directory and the
raw translator report is written as
`<temp-case>/crosstl-out/portability-report.json`. When `--reports-dir` is set,
the runner copies that report to
`<reports-dir>/<case>-portability-report.json` and writes the validation payload
to `<reports-dir>/<case>-validation.json`. Inspect the copied files because the
temporary case directory is removed after the run. In `--update` mode, the
checked-in `crosstl-out/` directory is regenerated, but the report is still
removed from the case after validation; use `--reports-dir` to keep it.

For reports whose recorded project root still exists, start with the bounded
text view when the raw report is too large to review directly:

```bash
python -m crosstl inspect-report \
  /tmp/crosstl-demo-reports/directx-shader-compiler-neg1-portability-report.json \
  --format text
```

Copied `--check` reports may emit that text and still exit nonzero after the
runner removes the temporary case directory, because freshness checks can no
longer find `project.root`, `project.config`, or external corpus files. Treat
those stale-root diagnostics as a consequence of demo cleanup; use the copied
JSON fields below and `<case>-validation.json` for the run-time result that was
computed before cleanup.

For direct JSON triage, the most useful fields in
`<case>-portability-report.json` are:

- `summary.diagnosticCounts`, `summary.diagnosticsByCode`,
  `summary.diagnosticsByTarget`, `summary.diagnosticsBySourceBackend`,
  `summary.diagnosticsByVariant`, and `summary.diagnosticsByCheckKind` group
  translation and validation diagnostics by severity, diagnostic code, target,
  source backend, variant, and check kind. Any nonzero `error` count explains a
  nonzero project translation exit.
- `diagnostics[]` contains the individual diagnostic messages and source or
  generated-artifact locations. Use the grouped summary first, then open the
  matching diagnostic records for the failing case.
- `summary.missingCapabilityCounts` groups unsupported source features,
  include/define processing gaps, provenance issues, artifact validation gaps,
  and optional toolchain-validation gaps.
- `artifactMatrix` and `artifacts[]` show the expected target/source/variant
  artifact plan and the emitted artifact records. Check `artifactMatrix` for
  missing, extra, or failed counts, then inspect `artifacts[].status`,
  `artifacts[].path`, `artifacts[].target`, `artifacts[].sourceBackend`,
  `artifacts[].sourceMap`, and `artifacts[].sourceRemap`.
- `validation.summary` and `validation.artifacts[]` record source and generated
  hash status, byte-size status, source-map status, source-remap status, and the
  final per-artifact validation `status`.
- `validation.toolchains[]` records target tool availability. `available` means
  the configured tools were found, `unavailable` means a configured tool is
  missing in the current environment, and `not-configured` means that target has
  no validation hook.
- `validation.toolchainRuns[]` appears when `--run-toolchains` is used. Each
  record includes `status`, `checkKind`, `source`, `path`, `target`,
  `sourceBackend`, `command`, `returncode`, `stdout`, and `stderr`. Failed runs
  are also reflected in diagnostics and in the validation report rollups.

The sibling `<case>-validation.json` is the `validate-project` JSON output. It
repeats the quick triage rollups under fields such as `success`,
`diagnosticCounts`, `missingCapabilityCounts`, `artifactStatusByTarget`,
`toolchainStatusCounts`, `toolchainRunStatusCounts`,
`toolchainRunStatusByTarget`, `toolchainRunStatusBySourceBackend`,
`toolchainRunStatusByCheckKind`, and `toolchainRunStatusByTool`. Use this file
for automation or a fast status check; use `<case>-portability-report.json` for
the full source, artifact, diagnostic, and validation detail.

CI keeps the same report files under `support/generated/demo-reports` and
uploads them from the `Open-Source Porting Demo` workflow as the artifact named
`open-source-porting-demo-reports-${{ matrix.os }}`. The reproducibility check
writes per-case reports under
`support/generated/demo-reports/${{ matrix.os }}/artifacts`, and platform smoke
checks write under `support/generated/demo-reports/linux-opengl`,
`support/generated/demo-reports/linux-vulkan`,
`support/generated/demo-reports/macos-metal`, or
`support/generated/demo-reports/windows-directx`. If the pytest selection fails,
CI also uploads `open-source-porting-demo-failure-summary-${{ matrix.os }}`
with the JUnit XML and failure-summary JSON/Markdown files.

## Third-party notices

The source slices remain under their upstream project licenses. See
`THIRD_PARTY_NOTICES.md` for repository, license, and source path details.
Exact pinned source URLs are recorded in each case's `corpus.json` manifest.
