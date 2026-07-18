# MLX Project Porting Integration

This directory contains the project-level MLX porting checks used by CrossTL.
The checks are pinned to MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` and exercise the Metal kernel tree
as a source repository, not as isolated parser snippets. This pinned revision is
an active repository-level verification target: configured coverage and expected
baselines are not, by themselves, evidence that every kernel translates or
passes a target validator.

## Scope

The current harness verifies:

- the [pinned native MLX Metal reference baseline](NATIVE_METAL.md) for exact
  upstream commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`. On the Arm64
  `macos-26` runner, it compiles all 40 native Metal units (33 Metal 3.2 / 7
  Metal 4.0), runs
  776 Python tests with 44 skips, and passes 260/260 C++ cases and 3,490/3,490
  assertions. This native upstream reference does not establish
  translated-target correctness, runtime parity, or numerical parity;
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
- a checked-in reduced Metal fixture that mirrors MLX's reference-returning
  `frag_at` accessor over `val_frags[i * width + j]`. The fixture is translated
  to DirectX and OpenGL through the public `translate-project` CLI and retains
  three separate source contracts. The mutable scalar receiver is declared
  through a function-local `using` alias and assigned through `frag_at`. An
  implicit const scalar call is passed directly to a read-only helper. A second
  outer value owns a `float2`-backed `nestedTile`; its const `store` method binds
  `thread const auto& accum = nestedTile.frag_at(i, j)` and reads `accum[k]`.
  Each generated target must assign the scalar sentinel directly to the original
  `val_frags[...]` lvalue, read back that exact element, lower the implicit const
  call to its backing storage, and replace the nested accessor and `accum` alias
  with a read from `self.nestedTile.val_frags[...][k]`. A value-return helper,
  retained alias, or copied tile does not satisfy the proof. Windows CI requires
  DXC compilation; Linux CI requires `glslangValidator` compilation and
  `spirv-val` validation for OpenGL 4.5. The macOS matrix leg verifies the same
  proof set in both generated artifacts without requiring either target
  compiler. These checks do not execute an MLX runtime or establish numerical
  parity;
- a separate checked-in Metal fixture for the MLX `BaseMMAFrag::load` call
  shape in which a templated tile method passes `&(src[index])` to a templated
  fragment helper. Project translation must emit both DirectX and OpenGL
  artifacts with zero diagnostics. The materialized fragment helper must keep
  a pointer-backed source view and read it at `stride`; a scalar `float src`
  parameter is rejected. The DirectX proof requires a `StructuredBuffer<float>`
  source and preserves the addressed `src[index]` position as a composed
  `src_offset + index` view. The OpenGL proof requires the equivalent global
  storage-buffer plus `src_offset` form, carries `index` into that offset, and
  reads `src[src_offset + stride]`. Source-style unresolved member calls are
  rejected. This gate inspects generated structure;
  it does not require native target compilation, execute a shader, or claim
  runtime parity;
- target-separated DirectX and Vulkan project runs for the same 11-source
  reduced frontier: `arange.metal`, `arg_reduce.metal`, `binary_two.metal`,
  `layer_norm.metal`, `logsumexp.metal`, `random.metal`, `rms_norm.metal`,
  `rope.metal`, `scaled_dot_product_attention.metal`, `softmax.metal`, and
  `ternary.metal`. Vulkan must translate and structurally validate all 11.
  DirectX emits the five sources whose aggregate entries do not require a
  runtime-selected workgroup size and records exact expected failures for the
  other six. Each blocked report must retain the pinned total specialization
  count and exactly match its diagnostic entry names to the materialized host
  names, with no additional diagnostics. Separate configs prevent DirectX
  workgroup contracts from being silently ignored by Vulkan, where project
  workgroup rules are unsupported.
  This establishes target-specific structural and toolchain coverage, not
  semantic readiness or runtime parity;
- a separate project-level expected-failure check for pinned `fence.metal`
  across DirectX, OpenGL, and Vulkan. Each target must report its exact
  `project.translate.*-atomic-fence-unsupported` diagnostic, target-specific
  `*.atomic-thread-fence-contract-lowering` missing capability, and requested
  `mem_device`, `memory_order_seq_cst`, `thread_scope_system` contract without
  emitting a target file. The blocked contract is tracked by
  [#1537](https://github.com/CrossGL/crosstl/issues/1537);
- materialization evidence for all 24 host-named `arg_reduce.metal` compute
  entries within 51 total specializations. Vulkan emits the aggregate artifact.
  DirectX and OpenGL fail before emission with
  `project.translate.workgroup-size-entry-ambiguous` because pinned host
  dispatch uses runtime axis and pipeline-limit operands unavailable to source
  materialization;
- DirectX HLSL compiler checks with official DXC v1.9.2602.24 on Windows CI for
  the emitted five-source frontier: `arange.metal`, `binary_two.metal`,
  `random.metal`, `rope.metal`, and `ternary.metal`. At the pinned revision the
  gate compiles every generated compute entry: 11, 225, 2, 18, and 212 entries
  respectively, for 468 generated compute entries in total. The
  pinned rope translation supplies required function constant IDs through the
  quoted `"1"`, `"2"`, and `"3"` selectors in
  `[project.specialization_constants]` and materializes the concrete DirectX
  variant before compilation. Aggregate conditional lowering
  completed under [#1695](https://github.com/CrossGL/crosstl/issues/1695) admits
  every pinned `ternary.metal` entry to this compiler gate. Target-ABI overload
  identity [#1694](https://github.com/CrossGL/crosstl/issues/1694) and
  minimum-precision arithmetic widening
  [#1701](https://github.com/CrossGL/crosstl/issues/1701) admit all 225
  `binary_two.metal` entries. Exact-layout DirectX union lowering
  [#1728](https://github.com/CrossGL/crosstl/issues/1728) admits both
  `random.metal` entries; broader union layouts remain tracked by
  [#1696](https://github.com/CrossGL/crosstl/issues/1696), and runtime dispatch
  metadata remains tracked by
  [#1542](https://github.com/CrossGL/crosstl/issues/1542). The six excluded
  aggregate sources cover 106 compute entries and are asserted as failed
  artifacts until their host dispatch contracts can be imported under
  [#1793](https://github.com/CrossGL/crosstl/issues/1793);
  no placeholder workgroup size is restored. `fence.metal` is
  excluded because its DirectX translation intentionally fails under
  [#1537](https://github.com/CrossGL/crosstl/issues/1537) before DXC. This gate
  establishes compiler acceptance only; it does not dispatch these kernels or
  establish numerical parity;
- a separate Windows CI Direct3D 12 execution proof for a checked-in,
  MLX-shaped Metal compute fixture with a file-scope immutable two-dimensional
  lookup table. The fixture is translated to HLSL during the test, compiled to
  DXIL with DXC, dispatched through the built-in DirectX runtime adapter and
  `compushady`, and read back as four exact unsigned values:
  `[5, 19, 11, 13]`. Because every result selects a table entry, this is a
  value-sensitive proof that the generated `static const` initializer survives;
  it is independent of the compiler-only frontier gate above;
- a second, independent Windows CI Direct3D 12 execution proof that translates
  the actual pinned `mlx/backend/metal/kernels/arange.metal` source with the MLX
  repository root as an include path. Project configuration selects the
  materialized `arangeuint32` entry and emits a standalone artifact at the
  deterministic `arange/arangeuint32.hlsl` path. The proof verifies the pinned
  source hash, entry-scoped provenance, source mapping, and the generated runtime
  artifact manifest before compiling `CSMain` to DXIL with DXC. It binds only
  the reflected `b0` start, `b1` step, and `u2` output resources and dispatches
  through the built-in Direct3D 12 adapter. Seven invocations use `start = 300`
  and `step = 17`; the required zero-tolerance readback is
  `[300, 317, 334, 351, 368, 385, 402]`;
- Vulkan assembly and validator checks for the existing non-fence regression
  frontier when SPIR-V tools are available. Vulkan atomic-fence feature work is
  deferred; the separate `fence.metal` contract check prevents generated
  barriers from being mistaken for semantic support;
- entry-scoped OpenGL packaging for the materialized `arangeuint32` compute
  entry from `arange.metal`. Project configuration selects the source entry,
  emits it at the deterministic `arange/arangeuint32.glsl` path as OpenGL
  `main`, and records the source-to-target entry identity in the portability
  report. The standalone artifact exposes only the `start`, `step`, and `out`
  resources and preserves the source arithmetic without an MLX source rewrite;
- an eight-source OpenGL frontier containing `arg_reduce.metal`,
  `binary_two.metal`, `logsumexp.metal`, `rms_norm.metal`, `rope.metal`,
  `scaled_dot_product_attention.metal`, `softmax.metal`, and `ternary.metal`
  in two project runs. `binary_two.metal`, `rope.metal`, and `ternary.metal`
  must emit with zero diagnostics and compile for OpenGL/SPIR-V 1.3 before
  `spirv-val`. Their project configuration supplies 24 source-qualified
  index-range assertions, and the portability report must reproduce the exact
  assertion count and content. The other five sources must produce the same
  exact fail-closed workgroup-size diagnostic and no target file. This is native
  artifact validation only; the gate does not run the kernels or establish
  runtime parity;
- full project materialization of pinned `gemv.metal` for DirectX. The gate
  requires 225 materialized specializations, no unsupported materializations or
  unresolved residue, no bare pure value-discard statements, one aggregate HLSL
  artifact, and exactly 224 host-named report execution entries joined by
  materialization identity. Every generated target entry and `numthreads`
  declaration must match its report contract. The emitted native 16-bit types
  require DXC to compile `CSMain`, `CSMain_85`, and `CSMain_113` under
  `cs_6_2` with `-enable-16bit-types` and zero diagnostics, then compile all
  224 functions in one `lib_6_6` invocation with the same flag, an exact
  export set, and exact profile-warning classification;
- full project materialization of pinned `gemv.metal` for OpenGL as a strict
  expected frontier. The project and report must retain the GEMV workgroup-size
  rule, and all 225 source specializations must materialize, after which
  translation must report the exact tracked workgroup-pointer diagnostic in one
  failed artifact record and emit no target file. This check performs no native
  validation or runtime execution;
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
- on Linux CI, native OpenGL 4.3 execution and readback for the selected
  `arangeuint32` artifact through ModernGL and Mesa EGL. Four invocations use
  `start = 300` and `step = 17`; the required zero-tolerance comparison is
  `[300, 317, 334, 351]`.
- on Linux CI, OpenGL SPIR-V specialization through PyOpenGL and Mesa EGL for a
  reduced generated compute artifact. The native adapter compiles the GLSL to
  SPIR-V, applies numeric constant ID 7, and requires exact readback for two
  independently selected unsigned values;
- on Windows CI, native Direct3D 12 execution and exact readback for both the
  reduced file-scope immutable lookup fixture and one generated uint32 entry
  from the pinned MLX `arange.metal` source. Neither proof executes the upstream
  MLX host runtime.

Pull requests run the 12-source pinned reduced scope: 11 non-fence frontier sources
and the explicitly blocked `fence.metal` contract source. They also run the
separate checked-in reference-accessor and template-member pointer fixtures;
those fixtures do not change the pinned MLX source count. Scheduled and
manually triggered CI also run the full-corpus artifact scout with finite Metal
template materialization budgets.
The generated full-corpus project config caps
`max_template_specializations` at 4096 and
`max_template_materialization_work` at 131072. The scout discovers all 40 pinned
MLX Metal kernel units and attempts 120 DirectX, OpenGL, and Vulkan artifacts.
Its fence-aware success condition is 117 translated artifacts and three expected
failed `fence.metal` records, one per target, with no fence target files emitted.
That condition is a gate expectation, not a claim that the pinned full corpus
currently satisfies it. Additional failures remain issue-backed scout results.
CI uploads generated portability reports, validation summaries embedded in
those reports, generated logs, available generated artifacts, and a concise JSON
summary.
Because `binary_two.metal` was already in the DirectX/Vulkan frontier, its OpenGL
promotion does not change either reduced source count.
The same applies to `ternary.metal`: its OpenGL promotion expands the native
toolchain gate without changing the 11-source non-fence reduced frontier.

The checked-in historical full-corpus scout snapshot against MLX revision
`968d264f2903d578e699c4452a4dbf48633921aa`
scanned 40 Metal kernels and attempted 120 target artifacts across DirectX,
OpenGL, and Vulkan. It translated 24 artifacts and reported 96 structured
artifact failures behind tracked issues. The current materialization pass
rejects template-hostile targets when concrete variants are missing instead of
emitting generic artifacts, so full-corpus counts should not be treated as
runtime-complete coverage or as the current fence-aware expected baseline.

This is shader/kernel artifact coverage. It does not claim that the MLX host
runtime has been ported to Direct3D, OpenGL, or Vulkan. Running the upstream MLX
GPU unit tests against non-Metal targets requires runtime adapters, host-side
dispatch wiring, data layout validation, and backend-specific build integration.
The reduced harness now emits runtime readiness artifacts so those gaps are
visible in CI reports without claiming runtime parity. The current readiness
manifests consume reflected runtime artifact metadata, including entry points,
resource bindings, and dispatch geometry. Runtime-test plans now resolve
source-level fixture names against common generated resource aliases. The
reference-accessor fixture is not included in those runtime manifests; its
scope ends at generated-source inspection and native compiler validation. The
reduced arange fixtures select the translated artifact by source and target,
then select `CSMain`, `main`, or `arangeuint32` independently for DirectX,
OpenGL, or Vulkan dispatch. OpenGL project translation packages source entry
`arangeuint32` as a standalone `main` artifact with an entry-scoped reflected
interface; its unsigned fixture uses values above the 8-bit range to detect
entry drift. The general DirectX readiness probe still describes the aggregate
artifact's first `uint8` entry. The separate required Windows device proof
packages `arangeuint32` as a standalone `CSMain` artifact. Vulkan execution selects
`arangeuint32`, `arangeint32`, and
`arangefloat32` explicitly and uses the same wide unsigned probe.
Plans report remaining non-blocking platform, layout, and entry-point ownership
warnings. The reduced fixture execution
report exercises the project runner and adapter contract with reference
buffers. The native execution report attempts the built-in native adapter
contract separately. On Windows CI the generated DirectX frontier HLSL must
compile with DXC. Separate native tests translate and execute the reduced
immutable lookup fixture and the pinned source's generated uint32 arange entry
through Direct3D 12. The arange test is numerical evidence for that one source,
entry, dtype, dispatch shape, and fixture only; it does not turn the frontier
compiler gate into a general runtime-parity claim. The five emitted DirectX
frontier artifacts carry exact per-source bfloat16 report evidence. All five
report `status=exact`, `approximationUsed=false`, a `uint-low-16-bits` register
representation, and round-to-nearest, ties-to-even conversion. All five emitted
sources require native `uint16` storage declarations and report the
`directx.native-16bit-types` capability. The harness compares each artifact's
`bfloat16Lowering` and `requiredCapabilities` fields with this pinned contract
and fails closed if either field is missing or changes. Native-profile bfloat
helpers now use exact `uint16_t` boundaries, and the two selected
`random.metal` entries compile without the promotion warnings tracked by
[#1799](https://github.com/CrossGL/crosstl/issues/1799). DXC reports zero
warnings across all 468 entry-point runs in the five-source emitted frontier.
The harness records this as a warning-clean contract and rejects any newly
observed warning. Contextual destination conversion under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) is resolved for the
pinned frontier. The arange assignment is emitted as
`arangeint16_out[index] = int16_t((uint(arangeint16_start) +
(index * uint(arangeint16_step))));`. The rope assignments are emitted as
`index_1 = uint(((2 * pos.x) + (pos.y * stride)));` and
`index_1 = uint((pos.x + (pos.y * stride)));`. All 18 rope entries compile with
DXC profile `cs_6_2`, `-enable-16bit-types`, and `-WX`. Native 16-bit arithmetic
conversion under [#1802](https://github.com/CrossGL/crosstl/issues/1802) is also
resolved for the pinned frontier. The float16 assignment is emitted as
`arangefloat16_out[index] = (arangefloat16_start + (float16_t(index) *
arangefloat16_step));`. All 11 arange entries compile without the
destination-conversion warning previously tracked by #1801 and with DXC profile
`cs_6_2`, `-enable-16bit-types`, and `-WX`. This preserves the resolved int16
destination-conversion evidence. The aggregate five-source frontier therefore
has compiler acceptance and a warning-clean diagnostic contract. This is
storage, conversion, report, and compiler evidence only; it does not execute a
bfloat16 workload or establish runtime or numerical parity. On macOS CI, the
generated `fence.metal` round-trip artifact must compile to AIR with the native
Metal compiler. This checks generated source and project metadata, not numerical
runtime parity or equivalent resource visibility. CrossGL/crosstl#1660 tracks
preservation of
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
git -C /tmp/mlx checkout 4367c73b60541ddd5a266ce4644fd93d20223b6e
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

On Linux, install the OpenGL, SPIR-V, and Vulkan runtime dependencies to require
the clean OpenGL compiler gates, the pinned GEMV expected frontier, and native
OpenGL and Vulkan execution of the generated MLX `arange` artifacts:

```bash
sudo apt-get update
sudo apt-get install -y glslang-tools libegl1 libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers spirv-tools vulkan-tools
python -m pip install moderngl==5.12.0 PyOpenGL==3.1.10 vulkan==1.3.275.1
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --require-opengl-frontier-toolchain \
  --require-opengl-gemv-frontier \
  --require-opengl-native-runtime \
  --require-vulkan-gemv-toolchain \
  --require-vulkan-toolchain \
  --require-vulkan-native-runtime

python demos/integrations/mlx/prove_copy_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/copy-opengl

python demos/integrations/mlx/prove_rms_norm_specialization.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/rms-norm-specialization \
  --require-opengl-toolchain
```

On Windows, install DXC to require DirectX HLSL validation for the reduced
frontier, the pinned GEMV compiler frontier, all three reference-accessor
artifact proofs, the selected LayerNorm entries, and the selected complex-copy
entry:

```bash
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root C:/path/to/mlx \
  --require-directx-toolchain \
  --require-directx-gemv-compiler-frontier

python demos/integrations/mlx/prove_layer_norm_directx.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/layer-norm-directx \
  --require-directx-toolchain

python demos/integrations/mlx/prove_copy_directx.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/copy-directx \
  --require-toolchain

python demos/integrations/mlx/prove_rms_norm_specialization.py \
  --mlx-root C:/path/to/mlx \
  --work-dir .crosstl-mlx-porting/rms-norm-specialization \
  --require-directx-toolchain
```

Install the DirectX runtime extra separately to execute the value-sensitive
lookup fixture through Direct3D 12:

```powershell
python -m pip install -e ".[directx-runtime]" pytest-xdist
$env:CROSTL_RUN_DIRECTX_LOOKUP_DEVICE_TEST = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_native_runtime_drivers.py `
  -k "directx_compute_runtime_executes_mlx_file_scope_lookup_on_device"
```

With the pinned MLX checkout available, run the generated arange proof
separately:

```powershell
$env:CROSTL_MLX_ROOT = "C:/path/to/mlx"
$env:CROSTL_RUN_DIRECTX_MLX_ARANGE_DEVICE_TEST = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_native_runtime_drivers.py `
  -k "directx_compute_runtime_executes_translated_pinned_mlx_arange_on_device"
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

The latest full-corpus scout at MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` discovered 40 Metal units, 841
include dependencies, and 120 planned target artifacts. It emitted `arange.metal`
for DirectX, OpenGL, and Vulkan plus a Vulkan `arg_reduce.metal` artifact before
the 900-second limit expired without a canonical project report. This interrupted
run does not establish an active full-corpus coordinate. CrossGL/crosstl#1376
tracks bounded materialization runtime and CrossGL/crosstl#1576 tracks durable
per-unit checkpointing. [#1676](https://github.com/CrossGL/crosstl/issues/1676)
remains the repository-level acceptance target.

Materialization charges configured work budgets to unique reachable entries,
helpers, struct specializations, and actual type-environment resolution. Exact
source-analysis snapshots and span indexes are reused with bounded retention;
the generated source remains unchanged. Artifact metadata keeps reachable
specializations, dependency-discovery work, and pruned eager candidates
separate.

A selected DirectX replay of `quantized.metal` now emits one artifact with zero
translation diagnostics for `affine_quantize_float_gs_32_b_2`. It materializes
six reachable specializations and three concrete records while pruning 110,861
unreachable candidates. The generated HLSL is 5,737 bytes with SHA-256
`c2737b9d324578209c15899cb9a1dad94697b041c0bcfd0c1276a809d36f8f88`.
This path verifies the completed template-member and owner-dependent `constexpr`
work tracked by CrossGL/crosstl#1476 and CrossGL/crosstl#1672. Native 16-bit HLSL
emission under
[#1799](https://github.com/CrossGL/crosstl/issues/1799) and concrete
`static_assert` evaluation under
[#1800](https://github.com/CrossGL/crosstl/issues/1800) are resolved for this
selected entry. The native-profile helper contract under #1799 is also resolved
across the pinned bfloat frontier. Contextual narrowing under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) remains recorded as
historical resolved evidence for this selected entry, while the pinned frontier
destination-conversion contract is resolved. The artifact requires the
`directx.native-16bit-types` capability and contains no remaining
`static_assert`. Official DXC validation with profile `cs_6_2`,
`-enable-16bit-types`, and `-WX` passes. The typed resource store is emitted as
`out_[uint((out_index + 4))] = uint(((output & 1095216660480ull) >> 32));`.
The locally generated DXIL was nonempty; its byte size is not treated as a
cross-version compiler invariant.

The adjacent DirectX entry `affine_gather_qmv_fast_float_gs_32_b_2` now
advances through the logical `static_assert` covered by
[#1800](https://github.com/CrossGL/crosstl/issues/1800). Its next one-unit
project record fails before artifact emission with
`project.translate.directx-private-pointer-unsupported` and missing capability
`directx.private-pointer-parameter-lowering`. The materialized helper
`load_vector_float_float_values_per_thread_2` receives the caller's
`thread U x_thread[values_per_thread]` array with `values_per_thread = 2`, but
the DirectX private-pointer analysis reports `missing-fixed-array-extent` for
parameter `x_thread`. This is tracked by the cross-target fixed-array alias
contract in [#1497](https://github.com/CrossGL/crosstl/issues/1497). No target
artifact is emitted, so native validation is not run for this entry.

A selected OpenGL replay of the same entry stops fail-closed with
`project.translate.opengl-index-type-unsupported` at `w[in_index + i]`. The
source index is `uint64_t`, the legal target index is `uint`, and the source
range is unproven. [#1515](https://github.com/CrossGL/crosstl/issues/1515)
tracks the generalized index-width normalization contract. No OpenGL target
artifact is emitted and native validation is not run. Neither selected replay
establishes runtime execution or numerical parity.

CrossGL/crosstl#1659 is complete; resource-register relocation no longer blocks
the selected aggregate DirectX replay. The checked-in evidence also records
full-kernel Vulkan replays of `fp_quantized.metal` and `quantized_nax.metal` as
failed after selected materialization contracts. Only the explicitly documented
reduced quantized fixtures carry validator evidence.
CrossGL/crosstl#1660 tracks resource coherence and volatility qualifiers that
are currently absent from Metal round-trip artifacts.
Native toolchain validation now runs on the target operating systems, completing
the coverage tracked by CrossGL/crosstl#1312. The bounded Direct3D and OpenGL
compute runtime drivers tracked by CrossGL/crosstl#1472 and
CrossGL/crosstl#1516 are also complete. CrossGL/crosstl#1388 still tracks the
artifact execution metadata required by runtime-test manifests and native
adapters, and CrossGL/crosstl#1462 covers wiring those contracts to general
device execution. OpenGL type aliases are inlined before host-interface
reflection and are not exposed as runtime resources. CrossGL/crosstl#1471
tracks entry-point ownership for reflected constants in runtime reports.
CrossGL/crosstl#1474 is represented by exact per-artifact DirectX
bfloat16 storage and conversion evidence in the pinned project report. The
harness fails closed unless each emitted source retains the exact
`bfloat16Lowering` object and corresponding `requiredCapabilities` list; this
does not extend the bounded runtime proof to bfloat16 or claim numerical parity.
Every custom DXC invocation derives its effective profile and compiler
arguments from the emitted HLSL through `crosstl.project.directx_toolchain`.
Generated `float16_t`, `int16_t`, or `uint16_t` types select at least Shader
Model 6.2 and add `-enable-16bit-types`; ordinary HLSL retains its selected
profile and command. These checks do not prove Direct3D 10 or 11 compatibility;
CrossGL/crosstl#1670 tracks explicit target profiles, feature gates, and
compiler selection. CrossGL/crosstl#1669 tracks
the fixed arrays of resource aliases introduced by the pinned revision's wide
quantized matrix-vector helpers. CrossGL/crosstl#1671 tracks workgroup backing
provenance through nested FFT helper parameters. A dedicated project replay now
translates the complete pinned `fft.metal` source for OpenGL with a 4,096
specialization limit and a 2,097,152-item materialization work budget. All 117
reachable template specializations materialize without unsupported residue, and
the diagnostic retains the `shared_in` backing object, zero element offset,
source parameter, and generated helper specialization for
`ReadWriter_float2_float2__load`. This advances beyond the earlier missing
concrete-backing failure. Translation now fails closed because the helper's
workgroup access range cannot be proven. No GLSL artifact is emitted, so
`glslangValidator`, runtime execution, and numerical parity do not apply to this
check yet. [#1671](https://github.com/CrossGL/crosstl/issues/1671) remains open
for the range-proof contract.

The pinned `gemv.metal` DirectX compiler frontier verifies source SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f` and
source size 5,383 bytes before translation. The project report must contain one
clean translated artifact, all 225 materializations with no unsupported
records, no unresolved materialization residue, and no standalone pure
value-discard statements such as `lid;`. Its generated SHA-256 and byte count
are checked against the emitted HLSL. Project configuration sets
`[project.workgroup_size_rules]` for `gemv.metal` to `[32, "BN", "BM"]`; the
report must retain the normalized `["32", "BN", "BM"]` rule. Exactly 224 of the
225 materializations must be host-named, and exactly 224 report execution
entries must join those records by `(hostName, materializedName)` identity. The
artifact must expose the exact target set `CSMain`, `CSMain_2`, ...,
`CSMain_224`, independently of report or materialization list order.

The resolved report sizes must be exactly `[32, 1, 1]`, `[32, 1, 4]`,
`[32, 1, 8]`, `[32, 2, 1]`, `[32, 4, 1]`, `[32, 8, 1]`, and `[32, 16, 1]`.
For every target entry, the emitted `numthreads` declaration must equal that
entry's report contract. This establishes exact workgroup-size specialization
for the generated aggregate artifact. DXC compiles representative scalar,
complex/Wave, and gather/constant-pointer paths (`CSMain`, `CSMain_85`, and
`CSMain_113`) with `cs_6_2` and `-enable-16bit-types`; all three invocations
must produce zero diagnostics. A second `lib_6_6` invocation retains
`-enable-16bit-types` while exporting and code-generating all 224 functions in
one DXIL library.

The library compile admits exactly 224 `numthreads ignored without accompanying
shader attribute` warnings caused by using a library profile. The gate derives
the expected warning source-line counts from the seven generated `numthreads`
forms and requires exact severity, message, source expression, and count
matches. Any unused-value warning, error, or other diagnostic fails the gate.
[#1786](https://github.com/CrossGL/crosstl/issues/1786) tracks the required
32-lane wave-size specialization. Library compilation proves that DXC accepts
and code-generates every exported function. It does not establish wave
semantics, runtime execution, numerical parity, or whole-kernel semantic
validity.

The separate pinned `gemv.metal` OpenGL frontier uses the same 4,096
specialization limit and 2,097,152-item materialization work budget. It requires
SHA-256 `c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f`,
one source unit, one failed artifact record, zero translated artifacts, zero
emitted target files, and all 225 specializations materialized with no
unsupported records. Its project configuration and report must retain the exact
`[32, "BN", "BM"]` workgroup-size rule, so workgroup-size configuration is not
an execution blocker. The required diagnostic is
`project.translate.opengl-workgroup-pointer-unsupported` with capability
`opengl.workgroup-pointer-lowering`; it must retain function
`GEMVKernel_bfloat16_t_1_8_1_32_1_4_false__run`, parameter and backing `tgp_memory`,
offset `0`, and reason `unprovable-view-access`. The concrete index derivation is
`sgN = simd_gid % 8`, `simdM = simd_gid / 8`, `bm = simdM`, and
`tgp_results = tgp_memory + sgN * 2 + bm`. Under the source-required 32-lane
subgroup width and the configured `[32, 8, 1]` workgroup, `simd_gid` is in
`[0, 7]`, the base offset is in `[0, 14]`, and the guarded reduction reaches at
most element `14` of the 16-element backing. Translation remains fail-closed
because the target-independent backing-view analysis does not yet carry this
range proof, while the target subgroup-width contract is also not established.
[#1671](https://github.com/CrossGL/crosstl/issues/1671) tracks backing and range
propagation and remains the translation blocker.
[#1786](https://github.com/CrossGL/crosstl/issues/1786) remains the execution
blocker for required subgroup-width specialization. Because translation emits
no GLSL artifact, this frontier does not claim an emitted or runnable GEMV
artifact, attempts no native compiler or runtime validation, and makes no
runtime or numerical-parity claim. Configuring the workgroup-size rule alone
does not prove the unavailable generated-artifact execution contract.

Owner-dependent `constexpr` helper calls in quantized struct static members now
resolve for the selected pinned replay, completing CrossGL/crosstl#1672.
CrossGL/crosstl#1491 tracks remaining qualified-static-constant materialization
outside the compiler-validated DirectX frontier.
Built-in overloads are resolved alongside user-defined wrappers by source
signature.
Before native validation, the harness verifies
the four numeric-to-Boolean SIMD wrapper conversions and the signed 8-, 16-,
32-, and 64-bit `arange` arithmetic conversions in the generated artifact.
Ubuntu CI installs `glslangValidator` and runs it with an OpenGL/SPIR-V 1.3
target. It first compiles the focused scalar-conversion fixtures successfully,
then compiles the translated `arange.metal` artifact. Shuffle-and-fill wrappers
lower through backend-neutral subgroup semantics and preserve their explicit
fill value for lanes below the delta.
The reduced DirectX/Vulkan frontier and the eight-source OpenGL artifact gate
include `scaled_dot_product_attention.metal`. Function-local scalar
and vector aliases now retain lexical scope and resolve across declarations,
constructors, casts, and generic static-member owners. For the pinned attention
source this resolves 531 concrete uses across all 42 entries, including the
float accumulation type used by the half and bfloat input families. The full
source translates to DirectX, OpenGL, and Vulkan; local OpenGL and Vulkan native
validation passes, and official DXC v1.9.2602.24 compiles all 42 generated
DirectX compute entries. This is compiler validation, not Direct3D runtime
execution or numerical parity.
The project Vulkan artifact is warning-free because project preparation removes
unreachable generic declarations. Direct single-file translation still emits
five such warnings under
[#1568](https://github.com/CrossGL/crosstl/issues/1568). Qualified pointer and
array aliases remain tracked in
[#1567](https://github.com/CrossGL/crosstl/issues/1567).
`arg_reduce.metal` materializes all 24 host-named entries. Vulkan emits the
aggregate artifact, while DirectX and OpenGL full-source packaging fails closed
with `project.translate.workgroup-size-entry-ambiguous` because the pinned host
selects axis and pipeline limits at runtime. Importing that dispatch contract is
tracked by [#1793](https://github.com/CrossGL/crosstl/issues/1793). Focused entry
lowering preserves address-space pointer provenance, fixed private-array extents,
and target dispatch dimensions, but does not establish an aggregate DirectX or
OpenGL artifact frontier. Entry-scoped packaging remains tracked in
[#1523](https://github.com/CrossGL/crosstl/issues/1523).
The reduced reference-accessor fixture covers three non-template paths. The
mutable scalar call returns the direct `val_frags[i * width + j]` lvalue and
must retain storage identity through assignment and readback. The implicit const
scalar call must lower directly into its read-only helper argument. The nested
const path matches the pinned `BlockMMA`/`Ctile` shape with a reduced outer value,
a `float2` fragment tile, a `thread const auto&` alias, and an `accum[k]` read.
For DirectX and OpenGL, the last path must contain neither `frag_at` nor `accum`
and must read `self.nestedTile.val_frags[...][k]`. This proof does not cover
template-indexed or nested forwarding overloads, full `quantized.metal`
translation, shader execution, numerical parity, or upstream MLX host/runtime
integration.
The reduced template-member pointer fixture covers the next `BaseMMAFrag::load`
boundary independently. It requires materialization of the generic
`SrcPtrType` helper from `&(src[index])`, a pointer-backed helper parameter or
equivalent OpenGL buffer-offset view, and an indexed `src[stride]` read whose
base offset still contains the outer `index`. It rejects a scalarized `float`
parameter even when no source-style call remains. The proof ends at artifact
structure and does not establish target compiler acceptance, shader execution,
or numerical parity.
`binary_two.metal` now also belongs to the required OpenGL toolchain frontier.
CrossTL commit `db593d19b` specializes fixed-array helper views to their concrete
runtime storage resources while retaining fixed extents and offsets. For the
pinned source, project translation emits zero diagnostics, the generated GLSL
compiles for OpenGL/SPIR-V 1.3, and the resulting SPIR-V passes `spirv-val`. This
resolves [#1661](https://github.com/CrossGL/crosstl/issues/1661) for the pinned
frontier. It is artifact and toolchain evidence only; it does not establish
numerical or runtime parity.
The clean OpenGL frontier supplies 24 configured index-range assertions, all with
inclusive bounds `[0, 2147483647]`. The expressions are `offset + i`, `a_idx`,
`b_idx`, `out_idx`, `out_idx++`, `idx.x`, and `idx.y` for `binary_two.metal`;
`batch_idx * offset_stride`, `freq_stride * pos.x`, `in_index_1`, `in_index_2`,
`out_index_1`, and `out_index_2` for `rope.metal`; and `offset + i`, `a_idx`,
`b_idx`, `c_idx`, `bidx`, `cidx`, `out_idx`, `out_idx++`, `idx.x`, `idx.y`, and
`idx.z` for `ternary.metal`. These records are explicit MLX host/runtime
portability preconditions for OpenGL. They are not inferred guarantees, CrossTL
does not enforce them at runtime, and they do not establish runtime integration
or numerical parity.
The eight-source OpenGL/SPIR-V gate includes `rms_norm.metal`, `rope.metal`, and
`scaled_dot_product_attention.metal`. Their Metal function constants retain
their numeric identifiers as native GLSL specialization constants; the gate
compiles each generated module for OpenGL/SPIR-V 1.3 and validates the resulting
binary. The native OpenGL runtime separately verifies typed specialization,
dispatch, and deterministic readback with a reduced generated artifact; it does
not claim numerical parity for those three full MLX kernels. For the pinned
DirectX rope check, project
configuration supplies IDs 1 through 3 and CrossTL materializes a concrete HLSL
variant before DXC.

The focused `prove_copy_opengl.py` gate translates the full upstream
`copy.metal` source pinned at commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` under one selected entry-point
scope, `s_copycomplex64float32`. The source declares 2,496 entries and expands
to 2,497 preprocessed instantiations. Exactly one selected specialization is
materialized, while the current evidence prunes 69,915 candidate pairs. No
generated wrapper fallback is used.

The generated OpenGL artifact must lower the source
`static_cast<float>(src[0])` conversion to one evaluation of `(src[0]).real`.
Linux CI compiles the resulting GLSL to OpenGL/SPIR-V 1.3 and validates the
module with `spirv-val`. This bounded lowering follows the pinned
`complex64_t` conversion body; generalized user-defined conversion operators
remain tracked by [#1744](https://github.com/CrossGL/crosstl/issues/1744). This
proof does not claim shader execution, numerical parity, runtime integration,
or passage of the MLX test suite.

The companion `prove_copy_directx.py` gate selects
`s_copycomplex64bfloat16` from the same full pinned `copy.metal` source. It
requires exactly one reachable specialization, verifies that each generated
store evaluates the complex source once and projects `.real`, and requires the
exact round-to-nearest-ties-to-even bfloat16 helper. Windows CI compiles the
standalone HLSL entry with DXC using `cs_6_2`, `-enable-16bit-types`, and
warnings as errors. This proves selected-entry translation and native compiler
acceptance; it does not execute the shader, establish numerical parity, port
the MLX runtime dispatch path, or run the MLX test suite.

The focused `prove_layer_norm_directx.py` gate translates two host-selected
single-row entries from the pinned `layer_norm.metal` source: forward float32
with axis size 4099 and VJP float32 with axis size 8192 and `has_w=true`. It
reads the exact host dispatch formulas from the pinned `normalization.cpp` blob
and checks that both workloads are exercised by the pinned MLX fast-operation
tests. Those inputs derive workgroup sizes `[544, 1, 1]` and `[1024, 1, 1]`.
Looped entries remain outside this proof because their workgroup sizes depend on
the selected Metal pipeline's `maxTotalThreadsPerThreadgroup()` value.

Both selected source templates declare `SIMD_SIZE = 32`, consume the Metal lane
and simdgroup index builtins, and call the shared `simd_sum` reduction path. The
DirectX project variants therefore require an exact subgroup-width rule of 32.
Each standalone HLSL artifact must retain the rule provenance, emit exactly one
`[WaveSize(32)]` beside its host-derived `numthreads` contract, and compile with
DXC using `cs_6_6` and warnings as errors. Windows CI applies that gate to both
entries. This is source identity, host-dispatch, translation, reflection, and
native compiler evidence. It does not execute either kernel, compare numerical
results, port the MLX host runtime, or claim coverage of the MLX test suite.

The checked-in
[`contracts/layer_norm.dispatch.json`](contracts/layer_norm.dispatch.json)
fixture contains exactly two pinned single-row float32 LayerNorm records from
MLX commit `4367c73b60541ddd5a266ce4644fd93d20223b6e`. It captures each
record's entry point, workgroup size, subgroup width, specialization constants,
and dispatch geometry together with the source and host-dispatch provenance.
The finite records cover the forward axis-size-4099 workload and the VJP
axis-size-8192 workload described above. Only the VJP record applies function
constant `20` (`has_w=true`), matching the pinned host dispatch; the forward
record carries no function constant. These records do not prove runtime
execution, numerical parity, looped variants, or the full MLX test suite.

Validate the fixture schema, provenance, deterministic identities, and bounded
evaluation with:

```bash
.venv/bin/python -m pytest -q -n auto \
  tests/test_mlx_dispatch_contract_fixture.py
```

The focused `prove_rms_norm_specialization.py` gate fixes the project-level
RMSNorm specialization contract to the same upstream commit and to
`rms_norm.metal` SHA-256
`5d411a2350ba7ddf84eb35f9dcac7cde0d441bd55fa1e9e1ccc61d490d428dee`.
It translates the upstream source through `crosstl.project.translate_project`.
The source check also requires all four kernel templates to retain
`constexpr int SIMD_SIZE = 32`, both simdgroup lane/group builtins, and all 12
`simd_sum` calls. This is a semantic input contract: compiling a target shader
without an exact 32-lane subgroup guarantee is not sufficient evidence for
these reductions.
The pinned MLX host computes single-row workgroup width as
`32 * ceil_div(ceil_div(axis_size, 4), 32)` and uses the selected pipeline's
`maxTotalThreadsPerThreadgroup` for looped kernels. The proof materializes
`[32, 1, 1]` and `[64, 1, 1]` as representative upstream-valid results of
those host formulas. These two sizes deliberately do not claim complete axis,
device-limit, or runtime-selected workgroup coverage.

For DirectX, two named project variants combine those workgroup sizes with the
required `has_w` function constant through both selector forms:
`has_w=false` by name at `[32, 1, 1]` and `"20"=true` by numeric ID at
`[64, 1, 1]`. The gate verifies variant selector and workgroup provenance,
concrete specialization materialization, the pinned source hash, and the
generated `static const bool has_w` value. The DirectX project configuration
sets `subgroup_width_rules["mlx/backend/metal/kernels/rms_norm.metal"] = 32`.
Each HLSL library artifact must retain the exact subgroup-rule provenance and
Shader Model 6.6 enforcement metadata for all 12 pinned host-named entries,
emit one `[WaveSize(32)]` and one matching `numthreads` attribute per entry,
and retain the reflected workgroup contract. Windows CI uses two
warning-as-error DXC runs to compile one reflected representative entry from
each HLSL library with `cs_6_6`; native 16-bit artifacts additionally pass
`-enable-16bit-types`.

For OpenGL, the `workgroup_32` and `workgroup_64` variants leave `has_w`
deferred, retain `layout(constant_id = 20)`, and split each host-named entry
into a standalone `main` artifact. The proof deliberately does not configure a
subgroup-width rule for OpenGL because this project contract cannot enforce an
exact subgroup width there. It requires subgroup provenance and enforcement
metadata to remain absent. Linux CI compiles all 24 GLSL artifacts to OpenGL
SPIR-V 1.3 and validates all 24 binaries with `spirv-val`; that result does not
establish the source's 32-lane simdgroup or `simd_sum` semantics.

This is translation and native compilation evidence only. It does not execute
RMSNorm, establish numerical or runtime parity, claim complete runtime
coverage, or claim support for the full MLX test suite. End-to-end device
execution and host selection of packaged variants remain tracked by
[CrossGL/crosstl#1462](https://github.com/CrossGL/crosstl/issues/1462) and
[CrossGL/crosstl#1735](https://github.com/CrossGL/crosstl/issues/1735). The
translated MLX `arange.metal` Direct3D numerical proof remains a separate
Windows CI check.

`fence.metal` emits no DirectX, OpenGL, or Vulkan target artifact. The harness
requires the target-specific structured diagnostics and the exact requested
atomic-fence operands under #1537 instead of accepting generated barrier text as
semantic evidence.
Future scouts should add issue-backed blockers only when there are
concrete repros. Host runtime integration gaps should be handled in repository
integration code or downstream runtime adapters, not hidden as shader
translation successes.
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

At pinned MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`, exact high-budget project
replays of the complete `fp_quantized.metal` source now advance past
`epilogue_op.apply` for both DirectX and OpenGL. The receiver declaration is
`thread const TransformNone_float_float& epilogue_op`. This frontier combines
helper array-decay deduction, specialized struct constexpr assertion evaluation,
lexical receiver alias resolution, statement-bounded member-template parsing,
concrete constructor preservation, line-wrapped qualified struct receiver
materialization, and contextual Metal method receiver resolution. CrossTL commit
`c7a3c61ad` resolves the contextual receiver on this path. Specialized struct
constexpr assertion evaluation resolves CrossGL/crosstl#1807.

Dependent helper deduction now resolves the function-local `BK_padded` and
`BN_padded` expressions together with the file-scope `SIMD_SIZE` constant before
specializing plain helper templates. Proven non-type arguments are serialized to
canonical values, so equivalent Boolean and integer spellings identify the same
concrete struct at the kernel call site and in the generated helper signature.
The exact DirectX and OpenGL runs each materialize 604 function
specializations with no unsupported template records. This advances the current
frontier through the applicable CrossGL/crosstl#1479 and CrossGL/crosstl#1490
contracts; both issues retain broader project-materialization scope.

Source-scoped project configuration now supplies concrete `true` values for
`align_M` (ID 200), `align_N` (ID 201), and `align_K` (ID 202) only to
`fp_quantized.metal`. Both target records preserve `project-source-pattern`
provenance. This advances the source-scoped configuration contract in
CrossGL/crosstl#1809 and the concrete function-constant contract in
CrossGL/crosstl#1538 without applying these identifiers to unrelated sources.

Both targets also advance through equivalent duplicate definitions of
`BaseMMAFrag_float_8_8::kFragRows` and through construction of
`QuantizedBlockLoader_float_32_32_36_1_64_16_4`. These paths exercise the
qualified static-constant contract in CrossGL/crosstl#1491 and the constructor
address-space provenance contract in CrossGL/crosstl#1810. Equivalent duplicate
owners now resolve `BaseMMAFrag_float_8_8::frag_type` to its concrete
two-component float vector, including component access at `k`; this resolves
CrossGL/crosstl#1811 for the pinned frontier. Constructor factories preserve the
`BlockLoader_float_16_32_36_1_64::src_ld` const-value initialization and lower
the partially initialized `MMATile_float_2_1_BaseMMAFrag_float_8_8::val_frags`
array through ordered element writes. These results advance the broader
constructor contracts in CrossGL/crosstl#1812 and CrossGL/crosstl#1813.

The complete materialized CrossGL intermediate reaches target generation. Strict
function-body parsing accepts the generic pointer reinterpretation in
`fp_qmv_wide_impl_bfloat16_t_16_4_2_16`,
`(vec<bfloat16_t, 4>*)(xv[v] + k0)`, including the generic pointee type. This
advances CrossGL/crosstl#1814 and removes
`project.translate.crossgl-function-body-parse-failed` from both exact project
runs.

Source whole-fragment reads and writes through `thread_elements()` references
are now canonicalized to ordered `cooperative_matrix_element` operations. The
resulting cooperative-matrix contract records the `metal_thread_elements`
layout, a 32-lane subgroup, two elements per lane, and
`metal_thread_elements_reference_view` provenance. These fields survive into
both target diagnostics instead of being inferred again after source lowering.
Reduced read and write helpers compile with the native Xcode Metal compiler.
This resolves CrossGL/crosstl#1815 and CrossGL/crosstl#1816 for the pinned
frontier.

The checked-in
[`contracts/cooperative-matrix-fragment-mapping.json`](contracts/cooperative-matrix-fragment-mapping.json)
contract records the concrete `tile_4x4_row_pair` mapping used by this pinned
MLX source. In `mlx/backend/metal/kernels/steel/gemm/mma.h`,
`BaseMMAFrag<T, 8, 8>::get_coord` defines `qid = lane / 4`,
`fm = (qid & 4) + ((lane / 2) % 4)`, and
`fn = (qid & 2) * 2 + (lane % 2) * 2`. The two lane elements therefore map to
`(fm, fn)` and `(fm, fn + 1)`. The contract contains the resulting coordinates
for all 32 lanes and records `mlx_steel_BaseMMAFrag_get_coord` provenance. This
is source-specific evidence for the pinned MLX specialization; it is not a
universal layout claim for Metal cooperative matrices.

The materialized CrossGL intermediate contains 16 source
`CooperativeMatrixType` contract nodes. Before the current contract-flow change,
two nodes carried the complete 12-field contract. In the verified replay, all 16
carry the `metal_thread_elements` layout, subgroup size 32, two elements per
lane, `metal_thread_elements_reference_view` provenance, the
`tile_4x4_row_pair` mapping, and `mlx_steel_BaseMMAFrag_get_coord` mapping
provenance.

Parsing creates eight `CooperativeMatrixOpNode` operations: seven `element`
operations and one `multiply_accumulate` operation. Each element operation now
has scalar `expression_type` `float` and intentionally has no matrix
`result_type`. The multiply-accumulate operation has complete cooperative-matrix
`result_type` and `expression_type` contracts that preserve the accumulator and
destination representation. Shared expression result inference therefore
resolves CrossGL/crosstl#1610 on this branch without claiming that scalar element
expressions require matrix result types.

DirectX and OpenGL now provide an explicit opt-in lane-local cooperative-matrix
software-lowering foundation for the exact registered 8-by-8, 32-lane,
two-elements-per-lane mapping. Reduced target tests compile and validate type
representation, element access, negation, and element-wise addition,
subtraction, and multiplication. Cooperative-matrix load, store, multiply, and
multiply-accumulate operations remain fail closed. The default behavior also
remains fail closed, and the option is not wired through project profiles or
configuration. Full software fallback, target policy, runtime execution, and
numerical parity remain unimplemented. CrossGL/crosstl#1602 and
CrossGL/crosstl#1820 remain open for that work.

Full pinned `fp_quantized.metal` replays now materialize the transitive local
`constexpr` chain that defines `values_per_thread`, including its fixed array
extents. This resolves the source-materialization contract in
CrossGL/crosstl#1824. Neither target stops at the former private-pointer boundary:
DirectX now reports
`project.translate.directx-cooperative-matrix-unsupported`, while OpenGL reports
`project.translate.opengl-cooperative-matrix-unsupported`. Both diagnostics
retain the `tile_4x4_row_pair` fragment mapping and
`mlx_steel_BaseMMAFrag_get_coord` provenance, and both fail closed before
artifact emission. These replays do not establish complete source translation,
runtime integration, runtime execution, or numerical parity.

The previously recorded pinned Vulkan replays confirmed that both affected
kernels advanced past this contract without producing a full artifact.
`fp_quantized.metal` then stopped at
type inference for the reference-returning `frag_at(i, j)` argument tracked in
CrossGL/crosstl#1557. `quantized_nax.metal` next stops because the dependent
static owner of `mma` is absent, so its empty tag argument has no selected
parameter type. Dependent static-owner materialization remains tracked in
CrossGL/crosstl#1574. These results establish translation-frontier progress
only; they do not include runtime integration or numerical parity.

The previously recorded full pinned Vulkan run advanced beyond the
generic-vector-width diagnostic. The contextual initializer contract
implemented for CrossGL/crosstl#1573 now rejects the empty
`metal::bool_constant<...>{}` argument instead of inferring a zero-length array.
The selected parameter type is still
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
CrossGL/crosstl#1672, CrossGL/crosstl#1659, CrossGL/crosstl#1516,
CrossGL/crosstl#1476, CrossGL/crosstl#1472, CrossGL/crosstl#1312,
CrossGL/crosstl#1661, CrossGL/crosstl#1573, CrossGL/crosstl#1555,
CrossGL/crosstl#1561,
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
CrossGL/crosstl#1661 is covered for pinned `binary_two.metal` by fixed-array
resource helper specialization in CrossTL commit `db593d19b` and the required
OpenGL/SPIR-V 1.3 compilation and validation gate.
CrossGL/crosstl#1807 is resolved for the pinned `fp_quantized.metal` frontier by
specialized struct constexpr assertion evaluation; contextual receiver
materialization remains tracked in CrossGL/crosstl#1479.
CrossGL/crosstl#1811 is resolved for the same frontier by equivalent duplicate
struct-alias resolution with concrete component typing and fail-closed conflict
diagnostics.
