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
| `sascha-willems-vulkan-conservative-triangle` | `SaschaWillems/Vulkan` at `2d16383d3121fb42b82d9aa3dc106a7f2a8f3ade` | MIT | GLSL | CrossGL, OpenGL, Metal, Vulkan | Uses the upstream conservative raster triangle vertex shader without semantic edits. DirectX semantic lowering remains outside this checked target subset. |
| `slang-hello-world-compute` | `shader-slang/slang` at `29e69b0bf626f87500be73a7fb3764db25658c66` | Apache-2.0 WITH LLVM-exception | Slang | CrossGL, OpenGL, Metal, DirectX, Vulkan | Uses the upstream compute shader unchanged. |

## Source adjustments

The DirectX, Metal performance, SaschaWillems triangle, and Slang cases keep
upstream source files unchanged apart from repository formatting checks. The
Vulkan Samples and Apple cases are reduced source slices copied from
fixture-backed upstream examples so the demo remains small and deterministic.
The reductions remove unrelated code around the shader construct being
demonstrated; they do not patch translator output.

The complete `lonelydevil/vulkan-tutorial-C-implementation` shader pair was
tested as a candidate and exposed an artifact path collision when `shader.vert`
and `shader.frag` are translated together. That translator issue is tracked in
issue #743, and the case is intentionally not checked in until project artifact
mapping preserves distinct stage files.

The `KhronosGroup/OpenCL-SDK` SAXPY kernel was also tested as a candidate and
exposed OpenCL lowering gaps for OpenGL and DirectX artifacts. That translator
issue is tracked in issue #751, and the case is intentionally not checked in
until the generated target shaders validate across the intended platform
matrix.

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
