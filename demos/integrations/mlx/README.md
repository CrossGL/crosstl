# MLX Project Porting Integration

This directory contains the project-level MLX porting checks used by CrossTL.
The checks are pinned to MLX commit
`968d264f2903d578e699c4452a4dbf48633921aa` and exercise the Metal kernel tree
as a source repository, not as isolated parser snippets.

## Scope

The current harness verifies:

- discovery of the MLX Metal kernel project surface under
  `mlx/backend/metal/kernels`;
- Metal-to-CrossGL-to-Metal translation of pinned `fence.metal`, including
  project artifact hashes, sizes, source maps, provenance, and native Metal
  compilation on macOS CI. The gate requires all three device-memory,
  sequentially consistent, system-scope atomic fences to survive the round
  trip without a weaker barrier fallback. Resource coherence and volatility
  preservation remain blocked by
  [#1660](https://github.com/CrossGL/crosstl/issues/1660), so this is not yet a
  complete semantic-equivalence claim;
- DirectX and Vulkan artifact generation for the 12-source clean reduced
  frontier: `arange.metal`, `arg_reduce.metal`, `binary_two.metal`, `fence.metal`,
  `layer_norm.metal`, `logsumexp.metal`, `random.metal`, `rms_norm.metal`,
  `rope.metal`, `scaled_dot_product_attention.metal`, `softmax.metal`, and
  `ternary.metal`;
- materialization of all 51 concrete `arg_reduce.metal` specializations and
  clean artifact generation for DirectX, OpenGL, and Vulkan. OpenGL compilation
  and both OpenGL-derived and native SPIR-V validation run in the required Linux
  toolchain gate;
- DirectX HLSL smoke checks with DXC on Windows CI for the verified subset
  (`arange.metal`, `arg_reduce.metal`, `fence.metal`, and `rope.metal`). The gate
  compiles every discovered entry, including the unsuffixed `CSMain`. Other
  clean frontier kernels translate to DirectX but are excluded for structural
  and semantic gaps recorded in `expected-gaps.json`;
- Vulkan assembly and validator checks when SPIR-V tools are available. The
  reduced frontier remains semantically blocked on the Metal atomic-fence order
  and scope contract tracked in
  [#1537](https://github.com/CrossGL/crosstl/issues/1537), even when every module
  passes structural validation;
- OpenGL artifact generation for `arange.metal`, including deterministic
  separation of the source `float` and `bfloat16_t` helper declarations after
  both map to GLSL `float` and source-typed call rewriting coverage;
- OpenGL artifact generation for the clean `arg_reduce.metal`,
  `logsumexp.metal`, and `softmax.metal` frontier. In the CI-required mode every
  artifact compiles for OpenGL/SPIR-V 1.3 and passes `spirv-val`;
- on Linux CI, full project materialization of `gemv.metal` to OpenGL followed
  by native GLSL compilation and SPIR-V 1.3 validation for all 225 source
  specializations represented by the generated artifact;
- on Linux CI, full project materialization and translation of `gemv.metal` to
  Vulkan produces 225 specializations and 224 `GLCompute` entry points. The
  generated artifact passes both `spirv-as` and `spirv-val` for `vulkan1.1`
  with zero semantic warnings and no known codegen fallbacks. This is structural
  validation, not numerical runtime parity;
- runtime artifact manifest, runtime-test manifest, and runtime-test plan
  generation for reduced `arange` readiness probes across DirectX, OpenGL, and
  Vulkan;
- reference runtime fixture execution reports for the reduced `arange`
  readiness probes, using supplied project test-runner adapters and
  deterministic expected-output checks;
- native runtime execution-readiness reports for the same reduced probes,
  using the built-in DirectX, OpenGL, and Vulkan native adapter contracts with
  missing runtime drivers reported as structured blockers;
- on Linux CI, native Vulkan execution and readback for the generated MLX
  `arange.metal` unsigned 32-bit, signed 32-bit, and floating-point entry points
  through the optional Vulkan compute runtime and Mesa Vulkan software driver.

Pull requests run the clean reduced frontier above. Scheduled and manually
triggered CI also run the full-corpus artifact scout with finite Metal template
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
source-level fixture names against common generated resource aliases. The
reduced arange fixtures select the translated artifact by source and target,
then select `CSMain`, `main`, or `arangeuint32` independently for DirectX,
OpenGL, or Vulkan dispatch. The aggregate DirectX and OpenGL artifacts currently
expose their first `uint8` specialization through `CSMain` and `main`; per-entry
target packaging remains tracked in CrossGL/crosstl#1523. Vulkan execution
selects `arangeuint32`, `arangeint32`, and `arangefloat32` explicitly, and its
unsigned fixture uses values above the 8-bit range to detect entry-point drift.
Plans report remaining non-blocking platform, layout, and entry-point ownership
warnings. The reduced fixture execution
report exercises the project runner and adapter contract with reference
buffers. The native execution report attempts the built-in native adapter
contract separately. On Windows CI the generated DirectX frontier HLSL must
compile with DXC. Current DirectX smoke artifacts lower MLX bfloat16 aliases to
HLSL `half` for toolchain coverage; exact bfloat16 storage and conversion
semantics remain tracked separately. On macOS CI, the generated `fence.metal`
round-trip artifact must compile to AIR with the native Metal compiler. This
checks generated source and project metadata, not numerical runtime parity or
equivalent resource visibility. CrossGL/crosstl#1660 tracks preservation of
the source `volatile coherent(system)` pointer contract. On
Linux CI the generated Vulkan
`arangeuint32`, `arangeint32`, and `arangefloat32` entry points must assemble,
load, dispatch, read back, and compare on the Vulkan compute runtime; other
unavailable native backends remain structured blockers until backend runtime
drivers are supplied by integration code. This still does not execute
the upstream MLX host runtime or the upstream MLX Python/C++ unit test suite on
non-Metal backends.

## Running Locally

Clone MLX and check out the pinned revision:

```bash
git clone https://github.com/ml-explore/mlx.git /tmp/mlx
git -C /tmp/mlx checkout 968d264f2903d578e699c4452a4dbf48633921aa
```

Run the project-porting harness from the CrossTL repository:

```bash
python demos/integrations/mlx/run_mlx_porting.py --mlx-root /tmp/mlx
```

Run the full-corpus artifact scout:

```bash
python demos/integrations/mlx/run_mlx_porting.py \
  --mode full-corpus \
  --mlx-root /tmp/mlx \
  --summary /tmp/mlx/.crosstl-mlx-porting/full-corpus-summary.json
```

On Linux, install SPIR-V tools and the Vulkan runtime dependencies to require
both Vulkan validation and native execution of the generated MLX `arange`
artifact:

```bash
sudo apt-get update
sudo apt-get install -y glslang-tools libvulkan1 mesa-vulkan-drivers spirv-tools vulkan-tools
python -m pip install vulkan==1.3.275.1
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --require-opengl-frontier-toolchain \
  --require-opengl-gemv-toolchain \
  --require-vulkan-gemv-toolchain \
  --require-vulkan-toolchain \
  --require-vulkan-native-runtime
```

On Windows, install DXC to require DirectX HLSL validation for the reduced
frontier:

```bash
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root C:/path/to/mlx \
  --require-directx-toolchain
```

On macOS, require native compilation of the generated Metal round-trip artifact:

```bash
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --require-metal-toolchain
```

The harness writes reports, generated artifacts, and command logs under
`<mlx-root>/.crosstl-mlx-porting`.

## Current Translator Gaps

CrossGL/crosstl#1376 tracks bounded runtime for the scheduled full-corpus scout.
CrossGL/crosstl#1659 tracks quadratic DirectX resource-register relocation for
large aggregate artifacts; a high-budget `binary.metal` DirectX translation
reaches that allocator after template materialization.
CrossGL/crosstl#1660 tracks resource coherence and volatility qualifiers that
are currently absent from Metal round-trip artifacts.
CrossGL/crosstl#1312 tracks native toolchain validation coverage for project
CI. CrossGL/crosstl#1388 tracks the artifact execution metadata required by
runtime-test manifests and native adapters. OpenGL type aliases are inlined
before host-interface reflection and are not exposed as runtime resources.
CrossGL/crosstl#1471 tracks entry-point
ownership for reflected constants in runtime reports. CrossGL/crosstl#1472
tracks the Direct3D compute runtime driver needed to move Windows DirectX
coverage from DXC validation to native dispatch/readback. CrossGL/crosstl#1474
tracks exact DirectX bfloat16 lowering beyond the current compile-time smoke
mapping. CrossGL/crosstl#1491 tracks the current scaled-attention
qualified-static-constant materialization blocker. Built-in overloads are
resolved alongside user-defined wrappers by source signature. Before native
validation, the harness verifies
the four numeric-to-Boolean SIMD wrapper conversions and the signed 8-, 16-,
32-, and 64-bit `arange` arithmetic conversions in the generated artifact.
Ubuntu CI installs `glslangValidator` and runs it with an OpenGL/SPIR-V 1.3
target. It first compiles the focused scalar-conversion fixtures successfully,
then compiles the translated `arange.metal` artifact. Shuffle-and-fill wrappers
lower through backend-neutral subgroup semantics and preserve their explicit
fill value for lanes below the delta.
The reduced DirectX/Vulkan frontier includes
`scaled_dot_product_attention.metal`; its OpenGL path remains excluded under
the function-constant contract in
[#1538](https://github.com/CrossGL/crosstl/issues/1538). Function-local scalar
and vector aliases now retain lexical scope and resolve across declarations,
constructors, casts, and generic static-member owners. For the pinned attention
source this resolves 444 concrete uses across all 36 entries, including the
float accumulation type used by the half and bfloat input families. The full
source translates to DirectX, OpenGL, and Vulkan; local OpenGL and Vulkan native
validation passes. DirectX remains outside the DXC gate for the independent
resource-cursor and fixed-array issues listed in `expected-gaps.json`.
The project Vulkan artifact is warning-free because project preparation removes
unreachable generic declarations. Direct single-file translation still emits
five such warnings under
[#1568](https://github.com/CrossGL/crosstl/issues/1568). Qualified pointer and
array aliases remain tracked in
[#1567](https://github.com/CrossGL/crosstl/issues/1567).
`arg_reduce.metal` now belongs to all three clean artifact frontiers.
Address-space pointer dereferences lower through retained resource provenance,
private pointer helpers use fixed local-array extents, and `threads_per_grid`
lowers to native GLSL dispatch dimensions or the generated DirectX dispatch-info
cbuffer. The aggregate OpenGL module typechecks every generated kernel body but
exposes only the first as `main`; entry-scoped packaging for the other 23 kernels
remains tracked in
[#1523](https://github.com/CrossGL/crosstl/issues/1523).
`rms_norm.metal` remains outside the OpenGL/SPIR-V gate until Metal function
constants are preserved as specialization inputs under
[#1538](https://github.com/CrossGL/crosstl/issues/1538).
The Vulkan `fence.metal` artifact retains deterministic evidence of the
unconsumed system-scope constant and is reported as structurally validated but
not semantically ready under #1537.
Future scouts should add issue-backed blockers only when there are
concrete repros. Host runtime integration gaps should be handled in repository
integration code or downstream runtime adapters, not hidden as shader
translation successes.
The full GEMV OpenGL gate accepts only the reserved double-underscore identifier
warnings tracked in CrossGL/crosstl#1513; any other native compiler warning
fails the check.
The full GEMV Vulkan gate materializes all 225 source specializations and emits
224 `GLCompute` entry points. The generated artifact passes both `spirv-as` and
`spirv-val` for `vulkan1.1`, with zero semantic warnings and no known codegen
fallbacks. This is structural validation only: runtime integration is not
included, and the result does not establish numerical runtime parity.

Read-only scalar storage-pointer reinterpretation now has a shared AST contract
and target lowering for DirectX, OpenGL, and Vulkan. A 32-bit scalar storage
resource can be viewed through aligned 8-, 16-, or 32-bit scalar elements;
source pointer offsets are converted to bytes before target indexing, and the
generated OpenGL and Vulkan artifacts pass native validators. Writable views,
64-bit backing layouts, and incompatible address-space or alignment cases remain
explicit diagnostics under CrossGL/crosstl#1546. Metal `dispatch_bool` callbacks
with one integral-constant parameter now lower to a runtime branch whose two
callback bodies retain distinct compile-time `true` and `false` values. Nested
dispatches expand the full Cartesian specialization and reduced DirectX,
OpenGL, and Vulkan project fixtures pass their native validators. Other callback
helpers remain tracked in CrossGL/crosstl#1554. Concrete `const_for_loop`
callbacks now expand in source order when all three bounds are integral, the
callback has reference capture and one `auto` parameter, and its body has no
callback-local control transfer. Expansion is enabled only when the source
defines the recognized `integral_constant`, `Int`, recursive loop, and arithmetic
operator contracts; unrelated helpers with the same names remain opaque. Nested
loops preserve exact
`integral_constant<int, N>` argument types; unresolved or unsafe callbacks remain
opaque and fail through the existing structured materialization path. A reduced
Vulkan fixture preserves four stores at indices `0`, `1`, `4`, and `5`, then
passes `spirv-as` and `spirv-val`. OpenGL expected-type propagation for the same
aggregate call arguments remains tracked in CrossGL/crosstl#1559.

An isolated high-budget `quantized_nax.metal` Vulkan project run now expands the
concrete NAX tile callbacks, resolves conditional function-local dimensions, and
materializes `NAXTile<T, BR, BC>` as concrete 2-by-2 specializations. Explicit
member-template binding now preserves `float16_t` threadgroup arrays, and the
template-hostile project path initializes the verified compile-time callback
contracts before member lowering. Bounded, non-variadic namespace-scope alias
templates are now canonicalized after callback and member lowering, then their
backing struct templates are materialized once more. The high-budget report no
longer contains any `Int<...>` use or `using Int` declaration and has fallen from
111 unsupported records to zero. Proven function-local integral constants now
feed inferred and explicit member-template arguments with lexical shadowing and
concrete `sizeof` handling, so `BK_padded` and `BN_padded` no longer create
symbolic helper specializations. Free helper deduction now retains unnamed
parameters, recognizes empty braced type values, and applies the same proven
lexical constants, which materializes `tile_matmad_nax` with concrete tile types
and transpose values. A verified `dispatch_bool` helper whose remaining reachable
calls are lambdas is handed to the existing callback lowering; named functors and
altered helper contracts retain ordinary materialization. Verified
`const_for_loop` callbacks now lower bare callback returns to per-iteration
escapes and fold bare integral-constant parameters only when the source defines
the verified implicit value conversion. Materialization completes with 722
specializations and no unsupported records.

Concrete struct-owned `using` and `typedef` aliases are now resolved inside
C-style and named cast targets after owner materialization. The rewrite respects
qualified owners, lexical shadowing, concrete float and integer specializations,
and aliases whose targets already contain pointer qualifiers. Metal cast nodes
retain source qualifiers while exposing a canonical target type to the strict
CrossGL function-body parser. Reduced DirectX, OpenGL, and Vulkan project
fixtures pass their native validators.

Concrete struct-owned alias templates now resolve their declaring owner,
default arguments, dependent owner constants, and alias chains before member
template deduction. Namespace-qualified and nested same-named owners remain
distinct, and generic vector locals retain their concrete type instead of
borrowing a later same-named declaration. Reduced four-component DirectX,
OpenGL, and Vulkan fixtures pass native validation. This is a partial
implementation of CrossGL/crosstl#1490; dependent function-local aliases and
value expressions outside this contract remain tracked there.

The isolated high-budget `quantized_nax.metal` run still completes 722
specializations with no unsupported records and resolves the NAX fragment
aliases to concrete eight-lane float, half, and bfloat vectors. Metal reverse
translation now represents those local values as fixed aggregate wrappers with
explicit lane storage and element-wise helpers. Reduced DirectX, OpenGL, and
Vulkan fixtures preserve lane reads, writes, arithmetic, and mutable helper
parameters; their generated artifacts pass the available native validators.
The lowering rejects unsupported operators, member selections, mixed vector
shapes, and ABI-visible device or constant storage instead of changing the
source contract. Direct generic-vector canonicalization outside the Metal
frontend remains tracked in CrossGL/crosstl#1569.

Generic member calls now retain their receiver, method, and ordered type and
value arguments in the shared AST, including pointer-member calls and nested
generic types. Metal materialization resolves concrete template methods on
direct and nested struct-field receivers before target generation. Reduced
fixtures containing the five-argument `Atile.load` and `Btile.load` forms from
`fp_quantized.metal` pass the available DirectX, OpenGL, and Vulkan validators.
Calls that reach a target without a concrete specialization fail with a
structured diagnostic instead of losing the generic suffix or computation.

Pinned Vulkan replays confirm that both affected kernels advance past this
contract without producing a full artifact. `fp_quantized.metal` next stops at
type inference for the reference-returning `frag_at(i, j)` argument tracked in
CrossGL/crosstl#1557. `quantized_nax.metal` next stops because the dependent
static owner of `mma` is absent, so its empty tag argument has no selected
parameter type. Dependent static-owner materialization remains tracked in
CrossGL/crosstl#1574. These results establish translation-frontier progress
only; they do not include runtime integration or numerical parity.

The full pinned Vulkan run now advances beyond the generic-vector-width
diagnostic. The contextual initializer contract implemented for
CrossGL/crosstl#1573 now rejects the empty `metal::bool_constant<...>{}` argument
instead of inferring a zero-length array. The selected parameter type is still
unavailable because the captured intermediate drops the dependent static owner
from `CTile::NAXFrag_t::mma`; CrossGL/crosstl#1574 tracks that remaining
materialization contract. The intermediate also retains unresolved
reference-returning `frag_at` calls, whose receiver identity remains tracked in
CrossGL/crosstl#1557. No full-kernel artifact or validator result is claimed.
Complete address-space, const, pointer-provenance, and
unresolved-alias diagnostic transport remains tracked in CrossGL/crosstl#1566.
Pointer-bearing aggregate propagation remains tracked in CrossGL/crosstl#1544,
and lowered receiver/reference semantics must satisfy CrossGL/crosstl#1557
before the kernel can be considered semantically ready.
Lazy logical and conditional evaluation in SPIR-V remains tracked in
CrossGL/crosstl#1560 for full-corpus semantic coverage.
Nested-return lowering in pointer-preserving SPIR-V inlining is covered by the
passing full GEMV Vulkan gate. Side-effectful compatibility arguments remain
rejected explicitly and tracked in CrossGL/crosstl#1562.

## Resolved Frontier Issues

The current reduced frontier no longer depends on the previously tracked issues:
CrossGL/crosstl#1573, CrossGL/crosstl#1555, CrossGL/crosstl#1561,
CrossGL/crosstl#1551,
CrossGL/crosstl#1498,
CrossGL/crosstl#1394,
CrossGL/crosstl#1317,
CrossGL/crosstl#939, CrossGL/crosstl#940,
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
CrossGL/crosstl#1354 and CrossGL/crosstl#1362 are closed by the current
full-corpus materialization and Vulkan validation work. CrossGL/crosstl#1452,
CrossGL/crosstl#1453, and CrossGL/crosstl#1454 are covered by bounded template
materialization with source-located diagnostics for unsupported MLX reduction,
scan, and Steel specializations. CrossGL/crosstl#1392 is closed by fixture
resource binding through reflected backend aliases. CrossGL/crosstl#1500 is
covered by mapped-signature collision detection with overload-aware GLSL call
rewriting. CrossGL/crosstl#1502 is covered by contextual GLSL aggregate
construction for struct, fixed-array, vector, and matrix values.
CrossGL/crosstl#1503 is covered by explicit expected-type scalar coercion for
numeric-to-Boolean returns and signed mixed-width `arange` arithmetic.
