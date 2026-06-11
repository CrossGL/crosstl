# MLX Project Porting Integration

This directory contains the project-level MLX porting checks used by CrossTL.
The checks are pinned to MLX commit
`968d264f2903d578e699c4452a4dbf48633921aa` and exercise the Metal kernel tree
as a source repository, not as isolated parser snippets.

## Scope

The current harness verifies:

- discovery of the MLX Metal kernel project surface under
  `mlx/backend/metal/kernels`;
- DirectX and Vulkan artifact generation for the current reduced frontier:
  `arange.metal`, `arg_reduce.metal`, `binary.metal`, `binary_two.metal`,
  `copy.metal`, `fence.metal`, `random.metal`, `rope.metal`, and
  `ternary.metal`, and `unary.metal`;
- Vulkan assembly validation when SPIR-V tools are available;
- OpenGL artifact generation for `arange.metal`.

A separate full-corpus scout against the same pinned MLX revision currently
translates 72 of 120 target artifacts across DirectX, OpenGL, and Vulkan before
the tracked gaps below.

This is shader/kernel artifact coverage. It does not claim that the MLX host
runtime has been ported to Direct3D, OpenGL, or Vulkan. Running the upstream MLX
GPU unit tests against non-Metal targets requires runtime adapters, host-side
dispatch wiring, data layout validation, and backend-specific build integration.

## Running Locally

Clone MLX and check out the pinned revision:

```bash
git clone https://github.com/ml-explore/mlx.git /tmp/mlx
git -C /tmp/mlx checkout 968d264f2903d578e699c4452a4dbf48633921aa
```

Run the project-porting harness from the CrossTL repository:

```bash
python integrations/mlx/run_mlx_porting.py --mlx-root /tmp/mlx
```

On Linux, install SPIR-V tools and require the Vulkan smoke check:

```bash
sudo apt-get update
sudo apt-get install -y spirv-tools
python integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --require-vulkan-toolchain
```

The harness writes reports, generated artifacts, and command logs under
`<mlx-root>/.crosstl-mlx-porting`.

## Current Translator Gaps

- CrossGL/crosstl#888: specialize template element types in generated OpenGL
  project artifacts.
- CrossGL/crosstl#889: lower Metal SIMD-group helper intrinsics for OpenGL
  validation.
- CrossGL/crosstl#890: replace DirectX `NoneType` crashes in MLX Steel kernels
  with translation or structured diagnostics.
- CrossGL/crosstl#891: allocate non-overlapping DirectX bindings for
  project-scale parameter blocks.
- CrossGL/crosstl#892: disambiguate promoted DirectX `groupshared` names for
  project kernels.
- CrossGL/crosstl#893: resolve overloaded storage-buffer helper calls for
  Vulkan project translation.

These gaps are translator work. Host runtime integration gaps should be handled
in MLX-specific integration code or downstream runtime adapters, not hidden as
shader translation successes.

## Runtime Integration Gaps

- CrossGL/crosstl#894: define a backend-agnostic runtime integration manifest
  for project translations.
- CrossGL/crosstl#895: add a cross-backend runtime verification harness for
  project translations, with MLX as the first substantial corpus.
