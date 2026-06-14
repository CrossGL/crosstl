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
  `arange.metal`, `binary_two.metal`, `fence.metal`, `random.metal`, and
  `ternary.metal`;
- Vulkan assembly and validator checks when SPIR-V tools are available and no
  active validation blocker is tracked;
- OpenGL artifact generation for `arange.metal`;
- optional full-corpus artifact generation for all 40 pinned MLX Metal kernels
  across DirectX, OpenGL, and Vulkan when `--full-corpus` is passed.

The default CI harness does not claim full-corpus success. The latest
issue-backed full-corpus scout against the same pinned MLX revision translated
57 of 120 target artifacts and reported 63 failed artifacts. CrossGL/crosstl#1354
tracks remaining project-scale Metal template materialization gaps. The previous
OpenGL template binding tracker, CrossGL/crosstl#1355, is closed and no longer
listed as an active MLX blocker.
Full-corpus pass states must come from a successful `--full-corpus` harness run,
not from stale metadata.

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

Run the optional full-corpus check when refreshing the MLX corpus status:

```bash
python integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --full-corpus \
  --require-vulkan-toolchain
```

Until CrossGL/crosstl#1354 is closed by fixes, the full-corpus check is expected
to fail instead of recording an unverified pass state.

The harness writes reports, generated artifacts, and command logs under
`<mlx-root>/.crosstl-mlx-porting`.

## Current Translator Gaps

The pinned reduced frontier emits all DirectX and Vulkan artifacts, and the
Vulkan SPIR-V artifacts validate with SPIR-V Tools. The optional full-corpus
check is blocked by CrossGL/crosstl#1354. CrossGL/crosstl#1362,
CrossGL/crosstl#1355, CrossGL/crosstl#1317, and CrossGL/crosstl#1300 are closed
or covered by current fixes and tracked with the resolved issues below.
Future scouts should add issue-backed blockers only when there are concrete
repros. Host runtime integration gaps should be handled in MLX-specific
integration code or downstream runtime adapters, not hidden as shader translation
successes.

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
current follow-up issue set. CrossGL/crosstl#1019, CrossGL/crosstl#1026,
CrossGL/crosstl#1028, CrossGL/crosstl#1029, CrossGL/crosstl#1030,
CrossGL/crosstl#1031, CrossGL/crosstl#1033, CrossGL/crosstl#1034,
CrossGL/crosstl#1035, and CrossGL/crosstl#1036 are closed by mainline fixes or
superseded by the current issue set and are no longer listed as active MLX
blockers. CrossGL/crosstl#1032, CrossGL/crosstl#1037, CrossGL/crosstl#1038,
CrossGL/crosstl#1039, CrossGL/crosstl#1068, CrossGL/crosstl#1104, and
CrossGL/crosstl#1105 are also closed or superseded by the current scout and
issue set. CrossGL/crosstl#1027 is no longer reported by the latest full-corpus
scout because the generated Metal quantization declarator now parses far enough
to reach target codegen. The current full-corpus scout no longer reports
runtime-adapter contracts, boolean SPIR-V interface lowering, or the previous
closed issue set as active missing capabilities. CrossGL/crosstl#1106,
CrossGL/crosstl#1107, CrossGL/crosstl#1110, CrossGL/crosstl#1111,
CrossGL/crosstl#1122, CrossGL/crosstl#1124, CrossGL/crosstl#1126, and
CrossGL/crosstl#1127 are also closed and are no longer tracked as active MLX
blockers. CrossGL/crosstl#852 is covered by the current OpenGL arange smoke
check. CrossGL/crosstl#1146 is resolved by bounded template replacement scans,
and CrossGL/crosstl#1184 is resolved by the latest mainline materialization
work. CrossGL/crosstl#1155 and CrossGL/crosstl#1160 are covered by the current
frontier after the SPIR-V project-artifact and multi-entry binding fixes.
CrossGL/crosstl#1203, CrossGL/crosstl#1204, and CrossGL/crosstl#1206 were
closed by the latest mainline helper-template, softmax parser, and SPIR-V
pointer-overload fixes. CrossGL/crosstl#1205, CrossGL/crosstl#1207,
CrossGL/crosstl#1218, and CrossGL/crosstl#1222 are also closed by the current
mainline OpenGL template, SIMD helper, steel attention diagnostic, and steel GEMM
materialization fixes. CrossGL/crosstl#1238, CrossGL/crosstl#1239, and
CrossGL/crosstl#1240 are closed by the assembled SPIR-V validation, complex
helper call, and fence initializer fixes. CrossGL/crosstl#1246,
CrossGL/crosstl#1248, CrossGL/crosstl#1249, CrossGL/crosstl#1250,
CrossGL/crosstl#1259, CrossGL/crosstl#1260, and CrossGL/crosstl#1261 are closed
by the current mainline access-chain index, materialization scalability,
templated functor, and Vulkan validation fixes. CrossGL/crosstl#1274 and
CrossGL/crosstl#1287 are closed by the current Vulkan complex helper validation
and full-corpus Metal template materialization fixes. CrossGL/crosstl#1362 and
CrossGL/crosstl#1317 are covered by the current reduced-frontier Vulkan
validation path.
