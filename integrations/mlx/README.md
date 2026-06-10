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

- CrossGL/crosstl#850: lower MLX Metal `max_total_threads_per_threadgroup`
  metadata for DirectX and OpenGL compute output.
- CrossGL/crosstl#851: scope MLX resource declarations per generated DirectX
  and OpenGL entry point.
- CrossGL/crosstl#852: disambiguate OpenGL resource bindings for multi-entry
  MLX kernels.
- CrossGL/crosstl#853: remove Metal include directives from generated OpenGL
  MLX artifacts before validation.
- CrossGL/crosstl#854: emit valid SPIR-V literals for MLX quantized
  bit-packing constants.
- CrossGL/crosstl#855: avoid recursion overflow translating MLX `sort.metal`
  to Vulkan.

These gaps are translator work. Host runtime integration gaps should be handled
in MLX-specific integration code or downstream runtime adapters, not hidden as
shader translation successes.
