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
  `arange.metal`, `binary.metal`, `binary_two.metal`, `copy.metal`,
  `fence.metal`, `random.metal`, `ternary.metal`, and `unary.metal`;
- Vulkan assembly validation when SPIR-V tools are available;
- OpenGL artifact generation for `arange.metal`.

A separate full-corpus scout against the same pinned MLX revision currently
translates 67 of 120 target artifacts across DirectX, OpenGL, and Vulkan:
DirectX translates 33 of 40 artifacts, OpenGL translates 4 of 40 artifacts, and
Vulkan translates 30 of 40 artifacts. The current materialization pass rejects
template-hostile targets when concrete variants are missing instead of emitting
generic artifacts, so the full-corpus count reflects stricter diagnostics rather
than a claim that previously translated artifacts were runtime-complete.

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

- CrossGL/crosstl#1019 tracks the remaining project template materialization
  blocker group.
- CrossGL/crosstl#1026 covers generated Metal project metadata.
- CrossGL/crosstl#1028 through CrossGL/crosstl#1032 split the remaining Metal
  materialization work into scan helpers, variadic debug helpers, reduction and
  normalization kernels, convolution/GEMV sizing templates, and rotary embedding
  templates.
- CrossGL/crosstl#1033 through CrossGL/crosstl#1035 and CrossGL/crosstl#1068
  cover OpenGL template binding propagation for elementwise kernels,
  reduction/attention kernels, nested Steel kernels, and quantized/FFT/masked
  GEMV kernels.
- CrossGL/crosstl#1036 and CrossGL/crosstl#1037 cover the remaining SPIR-V
  storage-buffer helper provenance and inlining failures.

These gaps are translator work. Host runtime integration gaps should be handled
in MLX-specific integration code or downstream runtime adapters, not hidden as
shader translation successes.

## Runtime Integration Gaps

- CrossGL/crosstl#1038: emit backend-agnostic runtime binding manifests for
  translated project artifacts.
- CrossGL/crosstl#1039: add project runtime parity executors for translated GPU
  artifacts.

## Resolved Frontier Issues

The current reduced frontier no longer depends on the previously tracked issues:
CrossGL/crosstl#939, CrossGL/crosstl#940, CrossGL/crosstl#941,
CrossGL/crosstl#943, CrossGL/crosstl#944, CrossGL/crosstl#945, and
CrossGL/crosstl#946. CrossGL/crosstl#979, CrossGL/crosstl#980,
CrossGL/crosstl#981, CrossGL/crosstl#982, CrossGL/crosstl#983,
CrossGL/crosstl#984, CrossGL/crosstl#985, CrossGL/crosstl#1001,
CrossGL/crosstl#1002, CrossGL/crosstl#1003, CrossGL/crosstl#1004,
CrossGL/crosstl#1006, CrossGL/crosstl#1007, CrossGL/crosstl#1012, and
CrossGL/crosstl#1013 are also covered by mainline fixes or superseded by the
current follow-up issue set. CrossGL/crosstl#1027 is no longer reported by the
latest full-corpus scout because the generated Metal quantization declarator now
parses far enough to reach target codegen. The current full-corpus scout no
longer reports runtime-adapter contracts, boolean SPIR-V interface lowering, or
the previous closed issue set as active missing capabilities.
