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
- runtime artifact manifest, runtime-test manifest, and runtime-test plan
  generation for reduced `arange` readiness probes across DirectX, OpenGL, and
  Vulkan.

Pull requests run the reduced frontier above. Scheduled and manually triggered
CI also run the full-corpus artifact scout with finite Metal template
materialization budgets. The generated full-corpus project config caps
`max_template_specializations` at 4096 and
`max_template_materialization_work` at 131072. The full scout translates all 40
pinned MLX Metal kernel units to DirectX, OpenGL, and Vulkan and requires 120
clean generated artifacts: 40 DirectX, 40 OpenGL, and 40 Vulkan. It uploads
generated portability reports, validation summaries embedded in those reports,
generated logs, generated artifacts, and a concise JSON summary.

The last completed full-corpus scout against the same pinned MLX revision
scanned 40 Metal kernels and attempted 120 target artifacts across DirectX,
OpenGL, and Vulkan. It translated 24 artifacts and reported 96 structured
artifact failures behind tracked issues. The current materialization pass
rejects template-hostile targets when concrete variants are missing instead of
emitting generic artifacts, so full-corpus counts should not be treated as
runtime-complete coverage.

This is shader/kernel artifact coverage. It does not claim that the MLX host
runtime has been ported to Direct3D, OpenGL, or Vulkan. Running the upstream MLX
GPU unit tests against non-Metal targets requires runtime adapters, host-side
dispatch wiring, data layout validation, and backend-specific build integration.
The reduced harness now emits runtime readiness artifacts so those gaps are
visible in CI reports without claiming runtime parity. The current readiness
manifests consume reflected runtime artifact metadata, including entry points,
resource bindings, and dispatch geometry. Runtime-test plans now resolve
source-level fixture names against common generated resource aliases and report
remaining non-blocking platform or layout warnings. These plans are still
metadata readiness artifacts; they do not execute the upstream MLX runtime on
non-Metal backends.

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

Run the full-corpus artifact scout:

```bash
python integrations/mlx/run_mlx_porting.py \
  --mode full-corpus \
  --mlx-root /tmp/mlx \
  --summary /tmp/mlx/.crosstl-mlx-porting/full-corpus-summary.json
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

CrossGL/crosstl#1362 tracks the remaining Vulkan SPIR-V validation failures for
the reduced DirectX/Vulkan frontier. CrossGL/crosstl#1354 tracks the remaining
full-corpus Metal template materialization work for backend artifacts.
CrossGL/crosstl#1376 tracks bounded runtime for the scheduled full-corpus scout.
CrossGL/crosstl#1312 tracks native toolchain validation coverage for project
CI. CrossGL/crosstl#1388 tracks the artifact execution metadata required by
runtime-test manifests and native adapters. CrossGL/crosstl#1392 tracks fixture
resource binding through reflected backend aliases. CrossGL/crosstl#1394 tracks
entry-point-scoped runtime adapter contracts for multi-entry artifacts.
CrossGL/crosstl#1396 tracks generated GLSL helper uniforms that reflection
currently reports as layout-incomplete runtime resources. Future scouts should
add issue-backed blockers only when there are concrete repros. Host runtime
integration gaps should be handled in repository integration code or downstream
runtime adapters, not hidden as shader translation successes.

## Resolved Frontier Issues

The current reduced frontier no longer depends on the previously tracked issues:
CrossGL/crosstl#1317, CrossGL/crosstl#939, CrossGL/crosstl#940,
CrossGL/crosstl#941, CrossGL/crosstl#943, CrossGL/crosstl#944,
CrossGL/crosstl#945, and CrossGL/crosstl#946. CrossGL/crosstl#979,
CrossGL/crosstl#980,
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
and full-corpus Metal template materialization fixes. CrossGL/crosstl#1329,
CrossGL/crosstl#1338, CrossGL/crosstl#1340, and CrossGL/crosstl#1346 are closed
by the current project-scale template and SPIR-V validation fixes.
CrossGL/crosstl#1355 is closed by the current OpenGL MLX template binding fix.
