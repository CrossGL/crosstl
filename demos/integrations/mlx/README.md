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
- selected-entry Metal-to-CrossGL-to-Metal translation of all 877 discovered
  current-pinned entries in ``unary.metal``: 183 each for ``v_``, ``v2_``,
  ``gn1_``, and ``gn4large_``, plus 145 ``vn_`` entries. The exact contract
  spans 37 operators and 20 input/output type pairs across bfloat16, Boolean,
  complex64, float16, float32, signed and unsigned integers, FP8 encode/decode,
  and complex projection. Reachability pruning keeps one selected operator and
  kernel, materializes the required gather index helper only for ``gn*``, and
  reflects either three vector resources or five gather resources. Source
  ``constant`` shape/stride provenance, a read-only ``device`` dimension
  reference, and postfix output indexing survive the round trip. A host-owned
  ``[1, 1, 1]`` dispatch contract is preserved, and macOS CI compiles every
  deterministic artifact to non-empty AIR. This is complete discovered unary
  Metal compiler/reflection coverage, not Metal numerical execution;
- selected-entry translation of all 877 discovered current-pinned
  `unary.metal` entries to OpenGL. The schema-v2
  `contracts/unary.opengl-translation.json` contract pins every standalone
  `main` artifact across the same five shapes, 37 operators, 20 type pairs,
  1,243 materializations, and 3,363 reflected resources, totaling 4,060,696
  generated GLSL bytes. Translation requires three explicit host/runtime
  index-range preconditions for `offset + i`, `out_idx++`, and `idx`; these are
  portability promises rather than inferred or runtime-enforced bounds. The
  lowering maps Metal `log10` through one-evaluation GLSL `log2`, preserves user
  overloads, and exposes the read-only `device const int& ndim` as a storage
  buffer whose scalar expression aliases element zero. Five required Linux CI
  shards compile every artifact for OpenGL/SPIR-V 1.3 with
  `glslangValidator` and validate every module with `spirv-val`. The gate
  requires 877 non-empty SPIR-V 1.3 modules. This is complete OpenGL
  translation, reflection, and native compiler coverage, not numerical
  execution or MLX host runtime redirection;
- selected-entry translation of all 877 discovered current-pinned
  `unary.metal` entries to DirectX. The schema-v2
  `contracts/unary.directx-translation.json` contract pins every standalone
  `CSMain` artifact across the same five shapes, 37 operators, 20 type pairs,
  and 1,243 materializations, with 3,912 reflected HLSL resources: three per
  `v_` and `vn_` artifact, four per `v2_` artifact, and six per gather artifact.
  Work-per-thread and gather forms include the generated `CrossGLDispatchInfo`
  dispatch cbuffer at bindings `b3` and `b0`, respectively. DirectX bfloat
  lowering covers the
  complete unary intrinsic family, while `acosh`, `asinh`, and `atanh` decode
  through portable float helpers before exact bfloat reconstruction. Explicit
  contextual casts preserve float-to-native-16 return and initializer narrowing
  without DXC `-Wconversion` diagnostics under warnings-fatal compilation. The
  read-only source `device const int& ndim` becomes a `StructuredBuffer<int>`
  scalar view at element zero, and source `out_idx++` remains postfix in HLSL.
  The same three explicit host/runtime index-range preconditions are required.
  Five required Windows CI shards compile every artifact with pinned DXC,
  source-derived `-enable-16bit-types`, warnings fatal, profile `cs_6_2`, and
  entry point `CSMain`; the gate requires 877 non-empty DXIL modules. Together
  with the OpenGL proof this closes complete discovered unary translation,
  reflection, and native compiler coverage on both targets, not numerical
  execution or MLX host runtime redirection;
- selected-entry Metal-to-CrossGL-to-Metal translation of all 4,122
  discovered current-pinned binary entries from ``binary.metal``. The complete
  contract spans 18 shapes and 11 concrete kernel templates, 24 operators, and
  25 input/output type pairs. It emits one kernel and one or two reachable
  materializations per artifact (6,026 exact materializations total), reflects
  each exact three- through seven-resource ABI, and preserves a host-owned
  ``[1, 1, 1]`` dispatch. Required macOS CI compiles every artifact with
  warnings fatal and requires 4,122 non-empty AIR outputs. This proves every
  discovered binary instantiation for Metal translation, reflection, and native
  compilation, not Metal numerical execution;
- selected-entry translation of all 4,122 discovered current-pinned
  `binary.metal` entries to OpenGL. The schema-v2
  `contracts/binary.opengl-translation.json` contract pins every standalone
  `main` artifact across all 18 shapes, 11 templates, 24 operators, 25 type
  pairs, 6,026 materializations, and 19,106 reflected resources, totaling
  16,276,504 generated GLSL bytes. Seven explicit
  host/runtime index-range preconditions bound `offset + i`, `a_idx`, `b_idx`,
  `out_idx`, `out_idx++`, `idx.x`, and `idx.y` to signed 32-bit OpenGL index
  space; they are portability promises, not inferred or runtime-enforced
  checks, and unproven wide indices still fail closed. Twenty-four required
  Linux CI shards retranslate every exact entry, verify deterministic identity,
  materialization, workgroup metadata, and reflected ABI, compile for
  OpenGL/SPIR-V 1.3 with `glslangValidator`, validate with `spirv-val`, and
  require 4,122 non-empty SPIR-V modules. This is complete binary OpenGL
  translation, reflection, and native compiler coverage, not numerical
  execution or MLX host runtime redirection;
- selected-entry translation of all 4,122 discovered current-pinned
  `binary.metal` entries to DirectX. The schema-v2
  `contracts/binary.directx-translation.json` contract pins every standalone
  `CSMain` artifact across all 18 shapes, 11 templates, 24 operators, 25 type
  pairs, and 6,026 materializations. Nine source shapes that consume
  `threads_per_grid` receive explicit `CrossGLDispatchInfo` interfaces, raising
  the exact aggregate reflection count to 21,248 resources across three-
  through eight-resource ABIs. Computed-result bfloat `ArcTan2`, `LogAddExp`,
  and `Power` paths expand to float and reconstruct with round-to-nearest-even;
  `Maximum` and `Minimum` expand only for comparison and return the selected
  original bfloat payload without requantization. All other unproven bfloat
  builtins remain fail-closed. The
  same seven explicit host/runtime index-range preconditions remain required.
  Twenty-four required Windows CI shards retranslate every exact entry, verify
  deterministic identity, materialization, `CSMain` workgroup metadata, and
  reflected ABI, then compile with checksum-pinned DXC using native 16-bit
  types and warnings fatal. The gate requires 4,122 non-empty DXIL modules.
  Together with the OpenGL and Metal contracts this closes complete binary
  translation, reflection, and native compiler coverage, not numerical
  execution or MLX host runtime redirection;
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
  DirectX emits five aggregate artifacts whose entries do not require a
  runtime-selected workgroup size, two entry-scoped `layer_norm.metal`
  artifacts, two entry-scoped `logsumexp.metal` artifacts, and 12 entry-scoped
  `rms_norm.metal` artifacts selected by checked-in host dispatch contracts. It
  records exact expected failures for the other three sources. Each blocked
  report must retain
  the pinned total specialization
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
  entries within 39 total specializations. Vulkan emits the aggregate artifact.
  DirectX and OpenGL fail before emission with
  `project.translate.workgroup-size-entry-ambiguous` because pinned host
  dispatch uses runtime axis and pipeline-limit operands unavailable to source
  materialization;
- DirectX HLSL compiler checks with official DXC v1.9.2602.24 on Windows CI for
  the 21-artifact frontier representing eight pinned sources: `arange.metal`,
  `binary_two.metal`, two bounded `layer_norm.metal` entries, two bounded
  `logsumexp.metal` entries, `random.metal`, 12 test-derived `rms_norm.metal`
  entries, `rope.metal`, and `ternary.metal`. At the pinned revision the gate
  compiles 11, 225, 2, 2, 2, 12, 18, and 212 entries respectively, for 484
  generated compute entries in total. Each LayerNorm, LogSumExp, and RMSNorm
  artifact is emitted independently with its host-derived workgroup size and
  exact subgroup width; specialization constants are retained where required.
  The
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
  [#1542](https://github.com/CrossGL/crosstl/issues/1542). Host dispatch contract
  import was completed under [#1793](https://github.com/CrossGL/crosstl/issues/1793).
  The three pending aggregate sources cover 76 compute entries. Those historical
  aggregate runs do not consume the later entry-scoped bounded contracts and
  remain asserted as failed artifacts; no placeholder workgroup size is
  restored. Separate bounded GEMV, arg-reduce, Softmax, and scaled-attention
  translation, package, and native-runtime proofs are documented below.
  `fence.metal` is
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
- Direct3D 12 and OpenGL 4.3 native-loader execution of the actual pinned
  `mlx/backend/metal/kernels/binary.metal` source. The project request selects
  `ss_Addfloat32`, packages one standalone target entry, and verifies the exact
  upstream source hash, entry-scoped provenance, reflected resource bindings,
  and scalar buffer layouts. Four invocations read `1.5` and `2.25` from the
  two source buffers and must return `[3.75, 3.75, 3.75, 3.75]` from the
  translated kernel. Windows runs the HLSL artifact through Direct3D 12; Linux
  runs the GLSL artifact through a surfaceless Mesa OpenGL context. This proves
  one real binary operator entry and does not claim coverage of the other
  materialized binary variants or the upstream MLX host runtime;
- the `arangeuint32` and `ss_Addfloat32` native-loader checks above, together
  with the scalar copy, float32 dot-product, bounded GEMV, bounded Softmax,
  bounded scaled-attention, and unary Square and ArcCos checks described below,
  use the
  current corpus at commit
  `846d176227a0ac13d2667e58d2bb68b322109ab0`. The broader 40-unit frontier and
  its remaining proofs retain their recorded historical revision until each
  source contract is remeasured;
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
- a separate bounded OpenGL dispatch-contract proof for the actual pinned
  `logsumexp.metal` source. The two float32 workloads emit standalone GLSL
  artifacts with workgroup sizes `[32, 1, 1]` and `[288, 1, 1]`, an exact
  subgroup width of 32, and preserved subgroup max and sum reductions. Linux CI
  compiles both artifacts to OpenGL SPIR-V 1.3 and validates both binaries. The
  runtime contract requires `GL_KHR_shader_subgroup` and rejects a device whose
  `GL_SUBGROUP_SIZE_KHR` value is not 32 before shader compilation or dispatch;
- full project materialization of pinned `gemv.metal` for DirectX. The gate
  requires 226 materialized specializations—224 host-named entries plus both
  reachable `elem_to_loc` helper overloads—no unsupported materializations or
  unresolved residue, no bare pure value-discard statements, one aggregate HLSL
  artifact, and exactly 224 host-named report execution entries joined by
  materialization identity. Every generated target entry and `numthreads`
  declaration must match its report contract. The emitted native 16-bit types
  require DXC to compile `CSMain`, `CSMain_85`, and `CSMain_113` under
  `cs_6_2` with `-enable-16bit-types` and zero diagnostics, then compile all
  224 functions in one `lib_6_6` invocation with the same flag, an exact
  export set, and exact profile-warning classification;
- full project materialization of pinned `gemv.metal` for OpenGL. The gate
  requires all 226 specializations materialized, all 224 entry-scoped GLSL
  artifacts emitted, exact workgroup and subgroup execution records, and the
  five explicit host index-range preconditions used for 64-bit source indices.
  Linux CI compiles every artifact with `glslangValidator` and validates every
  resulting SPIR-V 1.3 module with `spirv-val`. This historical aggregate gate
  does not establish runtime execution or numerical parity; a distinct bounded
  current-pinned native-runtime proof is documented below;
- on Linux CI, full project materialization and translation of `gemv.metal` to
  Vulkan produces 226 specializations and 224 `GLCompute` entry points. The
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
- on Windows CI, native Direct3D 12 execution and exact readback for the reduced
  file-scope immutable lookup fixture, one generated uint32 entry from pinned
  `arange.metal`, and one generated float addition entry from pinned
  `binary.metal`. Linux CI executes the same binary entry through OpenGL. None
  of these proofs executes the upstream MLX host runtime.

Pull requests run the 12-source pinned reduced scope: 11 non-fence frontier sources
and the explicitly blocked `fence.metal` contract source. They also run the
separate checked-in reference-accessor and template-member pointer fixtures;
those fixtures do not change the pinned MLX source count. Scheduled and
manually triggered CI also run the full-corpus artifact scout with finite Metal
template materialization budgets.
The generated full-corpus project config caps
`max_template_specializations` at 4096 and
`max_template_materialization_work` at 131072. Each artifact translation also
has a 120-second limit. A timed-out artifact is recorded with a structured
`project.translate.timeout` diagnostic, and the remaining canonical plan
continues. The outer 900-second command limit remains an aggregate CI boundary;
the durable checkpoint preserves every completed result before that boundary.
The scout discovers all 40 pinned MLX Metal kernel units and attempts 120
DirectX, OpenGL, and Vulkan artifacts.
Its fence-aware success condition is 117 translated artifacts and three expected
failed `fence.metal` records, one per target, with no fence target files emitted.
That condition is a gate expectation, not a claim that the pinned full corpus
currently satisfies it. Additional failures remain issue-backed scout results.
CI uploads generated portability reports, validation summaries embedded in
those reports, the durable full-corpus translation checkpoint, generated logs,
available generated artifacts, and a concise JSON summary. A subsequent
full-corpus invocation resumes verified completed jobs from a running or
interrupted checkpoint.
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
compiler gate into a general runtime-parity claim. The five aggregate DirectX
frontier artifacts carry exact per-source bfloat16 report evidence. All five
report `status=exact`, `approximationUsed=false`, a `uint-low-16-bits` register
representation, and round-to-nearest, ties-to-even conversion. All five
aggregate sources require native `uint16` storage declarations and report the
`directx.native-16bit-types` capability. The harness compares each artifact's
`bfloat16Lowering` and `requiredCapabilities` fields with this pinned contract
and fails closed if either field is missing or changes. Native-profile bfloat
helpers now use exact `uint16_t` boundaries, and the two selected
`random.metal` entries compile without the promotion warnings tracked by
[#1799](https://github.com/CrossGL/crosstl/issues/1799). DXC reports zero
warnings across all 484 entry-point runs in the 21-artifact emitted frontier.
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
destination-conversion evidence. The five-source aggregate portion therefore
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
the clean OpenGL compiler gates, the pinned GEMV toolchain gate, and native
OpenGL and Vulkan execution of the generated MLX `arange` artifacts:

```bash
sudo apt-get update
sudo apt-get install -y glslang-tools libegl1 libgl1-mesa-dri libglx-mesa0 mesa-vulkan-drivers spirv-tools vulkan-tools
python -m pip install moderngl==5.12.0 PyOpenGL==3.1.10 vulkan==1.3.275.1
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --require-opengl-frontier-toolchain \
  --require-opengl-gemv-toolchain \
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

python demos/integrations/mlx/prove_quantized_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/quantized-opengl \
  --require-opengl-toolchain

python demos/integrations/mlx/prove_quantized_opengl.py \
  --mlx-root /tmp/mlx \
  --work-dir .crosstl-mlx-porting/quantized-gather-opengl \
  --entry-point affine_gather_qmv_fast_float_gs_32_b_2 \
  --require-opengl-toolchain
```

On Windows, install DXC to require DirectX HLSL validation for the reduced
frontier, the selected 256-point FFT entry, the pinned GEMV compiler frontier,
all three reference-accessor artifact proofs, the selected LayerNorm entries,
and the selected complex-copy entry:

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

The selected 256-point FFT proof uses the same pinned checkout and executes the
translated artifact through the native loader on Direct3D 12 WARP:

```powershell
$env:CROSTL_MLX_ROOT = "C:/path/to/mlx"
$env:CROSTL_REQUIRE_MLX_FFT_DIRECTX_NATIVE_LOADER = "1"
python -m pytest -q -n auto `
  tests/test_translator/test_mlx_fft_native_loader.py::test_pinned_mlx_fft_executes_through_directx_native_loader
```

On macOS, require native compilation of the generated Metal round-trip artifact:

```bash
python demos/integrations/mlx/run_mlx_porting.py \
  --mlx-root /tmp/mlx \
  --require-metal-toolchain
```

The harness writes reports, generated artifacts, and command logs under
`<mlx-root>/.crosstl-mlx-porting`.

## Corpus Baselines

The reduced compiler and runtime proofs remain pinned to MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e`. Their checked-in source hashes,
dispatch contracts, generated artifacts, and numerical results describe that
exact reference revision and are not relabelled when the upstream corpus moves.

The scheduled full-corpus scout is pinned separately to current MLX commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. Entry discovery for that revision
records 42 Metal source units and 17,319 host-visible entries with no discovery
diagnostics. The scout plans 84 source-target coordinates across DirectX and
OpenGL, writes resumable progress and portability reports, and reports
unsupported constructs as structured failures. This corpus scan measures
translation coverage only; it does not claim MLX runtime integration or
numerical parity on either target.

## Current Translator Gaps

The latest full-corpus scout at MLX commit
`4367c73b60541ddd5a266ce4644fd93d20223b6e` discovered 40 Metal units, 841
include dependencies, and 120 planned target artifacts. It emitted `arange.metal`
for DirectX, OpenGL, and Vulkan plus a Vulkan `arg_reduce.metal` artifact before
the 900-second limit expired without a canonical project report. That recorded
attempt predates durable progress files and therefore does not establish an
active full-corpus coordinate. Current runs write an atomic checkpoint containing
the completed, active, and pending job coordinates, accumulated diagnostics, and
partial artifact matrix. A later invocation resumes only artifacts whose source,
generated output, and source-remap identities still match. CrossGL/crosstl#1376
continues to track bounded materialization runtime.
[#1676](https://github.com/CrossGL/crosstl/issues/1676) remains the
repository-level acceptance target.

A bounded checkpoint probe against the same pin planned 80 DirectX and OpenGL
jobs. It preserved four completed records, identified `binary.metal`/DirectX as
active with 75 jobs pending, and validated the checkpoint after a bare process
interrupt. Resuming the probe retained those four records and returned directly
to the active coordinate. This verifies interruption recovery, not completion
of the full corpus or numerical parity.

Materialization charges configured work budgets to unique reachable entries,
helpers, struct specializations, and actual type-environment resolution. Exact
source-analysis snapshots and span indexes are reused with bounded retention;
the generated source remains unchanged. Artifact metadata keeps reachable
specializations, dependency-discovery work, and pruned eager candidates
separate.

A selected DirectX replay of `quantized.metal` now emits one artifact with zero
translation diagnostics for `affine_quantize_float_gs_32_b_2`. It materializes
six reachable specializations and three concrete records while pruning 110,861
unreachable candidates. The generated HLSL is 4,357 bytes with SHA-256
`a0f1a10def581f30dc34ed870b9ce36f70fb12abfd447e9b1b369524efde7438`.
This path verifies the completed template-member and owner-dependent `constexpr`
work tracked by CrossGL/crosstl#1476 and CrossGL/crosstl#1672. After unreachable
materializations are pruned, this selected float specialization contains no live
native-width 16-bit types, and its report records `requiredCapabilities=[]`.
Native 16-bit HLSL support under
[#1799](https://github.com/CrossGL/crosstl/issues/1799) remains resolved and
validated elsewhere, including the pinned bfloat frontier, but is not required
by this selected specialization. Concrete `static_assert` evaluation under
[#1800](https://github.com/CrossGL/crosstl/issues/1800) is resolved for this
selected entry. Contextual narrowing under
[#1801](https://github.com/CrossGL/crosstl/issues/1801) remains recorded in the
broader DirectX toolchain evidence, but this selected `bits = 2` entry resolves
`OutType` to source `uint32_t` and generated HLSL `uint`. Its final typed
resource store therefore needs no width conversion and is emitted as
`out_[uint((out_index / writes_per_reduce))] = output;`. The artifact contains
no remaining `static_assert`. Its ordinary type contract needs no native-16-bit
profile uplift, while the generated wave intrinsics keep the configured project
and compiler target scoped to `directx-12`. Official DXC validation with profile
`cs_6_0`, no `-enable-16bit-types`, and `-WX` passes.
The locally generated DXIL was nonempty; its byte size is not treated as a
cross-version compiler invariant. This evidence covers translation and compiler
acceptance only; it does not claim runtime execution or numerical parity.

The adjacent DirectX entry `affine_gather_qmv_fast_float_gs_32_b_2` also emits
one artifact with zero translation diagnostics. It materializes 11 reachable
specializations and eight concrete records while pruning 110,861 unreachable
candidates. The specialized `load_vector_float_float_16_2` helper retains the
caller's `thread U x_thread[values_per_thread]` storage with
`values_per_thread = 16`; HLSL represents it as an `inout float[16]` plus a
base offset and proves all four writes in each `i += 4` iteration. The
`qdot_float_16_2` helper retains the read-only `uint8_t` view over its
`uint32_t` storage resource. Each byte read selects the backing word and lane,
and the helper call composes the root word offset, byte-view offset, row offset,
and local alias offset without treating bytes as typed 32-bit elements.

The gather path also materializes the overloaded `elem_to_loc` helper as
`elem_to_loc_uint32_t`, preserving the source `uint32_t` index type. Its HLSL
call carries the independent shape and stride resource offsets, and no
unresolved `elem_to_loc(...)` call remains in the artifact.

The source `adjust_matrix_offsets` helper receives five storage pointers by
mutable reference. DirectX keeps each resource handle unchanged and passes its
logical offset as `inout int64_t`. The entry owns the offset variables, the
helper updates them, and the subsequent `qmv_fast_impl_float_32_2` call consumes
all five updated values. This preserves source pointer rebasing without passing
resource handles by reference or discarding writes to by-value offsets.

The pinned `gather_qmv` host dispatch in `mlx/backend/metal/quantized.cpp` sets
`bk = 32` and `MTL::Size group_dims(bk, 2, 1)`. The project rule therefore
emits `[numthreads(32, 2, 1)]`. The kernel's simdgroup indices require a
32-lane subgroup, so the generated HLSL also emits `[WaveSize(32)]` and requires
Shader Model 6.6. The resulting artifact is 16,359 bytes with SHA-256
`c64564b5705aa9ef16769c0d0ffda26a8852399d63460079cd449fe71323b5de`.
Windows CI compiles it with DXC profile `cs_6_6` and `-WX`. This is selected-entry
evidence for the fixed-array alias work tracked by
[#1497](https://github.com/CrossGL/crosstl/issues/1497) and the read-only storage
view work tracked by [#1546](https://github.com/CrossGL/crosstl/issues/1546),
the resource pointer-offset contract tracked by
[#1518](https://github.com/CrossGL/crosstl/issues/1518),
with the broader subgroup contract tracked by
[#1894](https://github.com/CrossGL/crosstl/issues/1894). Issue
[#1786](https://github.com/CrossGL/crosstl/issues/1786) established the exact
subgroup-width metadata and DirectX enforcement model. The remaining issues
cover broader cross-target, writable-view, alignment, subgroup fallback, and
runtime acceptance criteria. This proof does not dispatch the kernel through
Direct3D, run MLX tests, or claim numerical parity.

The current-corpus `affine_quantize_float_gs_32_b_2` entry now has a
selected OpenGL native-loader proof at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. The project configuration supplies
three source-qualified index-range assertions for `in_index + i`, `gindex`, and
`out_index / writes_per_reduce`, each with inclusive bounds
`[0, 2147483647]`. These are explicit host/runtime portability preconditions
used to justify legal GLSL `uint` subscripts; they are not inferred or enforced
at runtime.

The target-scoped Metal option
`project.source_options.metal.target_options.opengl.software_subgroup_width = 32`
selects a fail-closed software implementation of this entry's scalar
`WaveActiveMin(float)`, `WaveActiveMax(float)`, and
`WaveShuffleDown(uint, int)` operations. This selected proof uses one compute
entry with local size `[32, 1, 1]`. The bounded mode can also partition a
concrete workgroup of at most 1,024 invocations into multiple 32-lane logical
subgroups when local size X is divisible by 32; unsupported operations, payload
types, helper ownership, raw subgroup builtins, or unproven lane-dependent
control flow still reject translation. The default hardware KHR subgroup path
is unchanged.
The software artifact uses collision-safe shared scratch storage and eight
barriers, emits `CROSSTL_SOFTWARE_SUBGROUP_WIDTH`, and emits no KHR subgroup
extension, `gl_Subgroup*` use, `CROSSTL_REQUIRED_SUBGROUP_WIDTH`, or hardware
subgroup-width runtime metadata.

The exact generated GLSL is 6,642 bytes with SHA-256
`e4d8e5931bfc93f81e2c3686c102a1d676c9a3dcdfd6447e90918aa7581beecb`.
Materialization records three concrete and nine reachable specializations, zero
dependency-discovery work, and 104,702 pruned candidates. Linux CI compiles it
for OpenGL/SPIR-V 1.3 with `glslangValidator`; `spirv-val` accepts the module,
whose disassembly contains eight control barriers and no group-nonuniform
instruction. The runtime package reflects `wBuffer` at binding 0 and the
`out_Buffer`, `scalesBuffer`, and `biasesBuffer` read-write resources at
bindings 1 through 3. Through surfaceless EGL and Mesa software OpenGL, one
32-thread dispatch transforms `[0, 1, 2, 3]` repeated eight times into eight
packed `uint32` values of `27`, scale `-1`, and bias `3`.

This is exact translation, packaging, native execution, and numerical parity
for that one deterministic affine-quantize workload. It does not redirect the
MLX host runtime, run MLX's test suite, cover other quantized entries, or turn
the constrained software subgroup mode into a general divergent-subgroup
implementation. [#1515](https://github.com/CrossGL/crosstl/issues/1515) records
the completed index-width normalization contract, while
[#1894](https://github.com/CrossGL/crosstl/issues/1894) continues to track
broader subgroup fallback coverage.

The adjacent `affine_gather_qmv_fast_float_gs_32_b_2` entry now has the same
selected-entry OpenGL gate. It emits a 16,132-byte GLSL artifact with SHA-256
`d765c7694d32be8a0cd31c0250a7ff7839e9fb2e11da9cb470344d16669ec8a6`, zero
project diagnostics, 11 reachable specializations, and eight concrete
materializations. The project report retains the pinned `32 x 2 x 1`
workgroup rule and four explicit index-range preconditions for the gather
resource lookups. The generated `load_vector` and `qdot` helpers preserve the
16-element private array. Reinterpreted storage pointers pass their root byte
base separately from their logical byte-view offset, so `qdot` composes the
incoming word offset, row offset, and loop index without capturing caller-local
expressions or scaling the word offset twice. Linux CI compiles the artifact for
OpenGL/SPIR-V 1.3 and validates the module with `spirv-val`. This advances the
cross-target contracts tracked by [#1497](https://github.com/CrossGL/crosstl/issues/1497),
[#1518](https://github.com/CrossGL/crosstl/issues/1518), and
[#1546](https://github.com/CrossGL/crosstl/issues/1546); their broader acceptance
criteria remain open. This remains compile-only evidence and does not claim
OpenGL dispatch, MLX test execution, or numerical parity.

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
CrossGL/crosstl#1516 are also complete. Packaged artifacts can now be dispatched
through the native-loader bridge; remaining source-specific execution gaps are
recorded with the individual proofs below. CrossGL/crosstl#1388 still tracks the
broader artifact execution metadata contract. OpenGL type aliases are inlined
before host-interface reflection and are not exposed as runtime resources.
CrossGL/crosstl#1471
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
quantized matrix-vector helpers. CrossGL/crosstl#1671 originally tracked
workgroup backing provenance through nested FFT helper parameters. A dedicated
DirectX project replay now selects the forward complex 256-point entry
`fft_mem_256_float2_float2` from the complete pinned `fft.metal` source. The
host planner's `MTL::Size(1, threadgroup_batch_size, threads_per_fft)` dispatch
maps to `[numthreads(1, 1, 64)]`; the report and generated HLSL must retain that
axis order. The replay materializes 24 reachable template specializations and
21 of the 22 configured function constants. Function constant 3 is pruned
because its Rader transform branch is unreachable for this power-of-two plan.
The generated artifact contains no first-class workgroup pointer residue and is
locked by SHA-256 and byte count. Windows CI compiles it with DXC using
`cs_6_2`, `-enable-16bit-types`, and warnings as errors. It then packages and
reflects the artifact, requiring 8-byte `float2` strides for the input and
output resources and a 16-byte constant-buffer allocation for the generated
`uint3` dispatch input. The native-loader request derives that input from the
physical workgroup count before Direct3D 12 WARP dispatch and readback.

The runtime check supplies an index-1 complex unit impulse and compares all 256
complex outputs with the analytical forward DFT unit circle at `2e-4` absolute
and relative tolerance. This verifies one bounded workload through translation,
packaging, native execution, and numerical comparison. It does not redirect the
MLX host runtime, run the MLX test suite, or establish broad FFT or backend
parity. The aggregate DirectX source materialization limit remains tracked by
[#1916](https://github.com/CrossGL/crosstl/issues/1916); broader runtime grid and
resource-layout contracts remain tracked by
[#1542](https://github.com/CrossGL/crosstl/issues/1542) and
[#1543](https://github.com/CrossGL/crosstl/issues/1543).

That bounded runtime proof now also covers current corpus commit
`846d176227a0ac13d266ce4644fd93d20223b6e`, where MLX generalizes FFT
workgroup storage through `FFTIOTypeTraits` and adds function constant 22 for
the Bluestein twiddle-table path. Replaying the same 256-point entry with that
constant set to `false` materializes 37 specializations, records 42 reachable
specializations before pruning, and concretizes 21 of 23 configured function
constants. The source reaches two compatible `ReadWriter_float2_float2__load`
overload bodies. DirectX workgroup-pointer reachability now analyzes both
bodies, merges their forwarded calls and access ranges, and retains the
concrete `fft_mem_256_float2_float2_shared_in[256]` backing identity. Generated
helpers transport `crosstl_ptr_buf` as an integer offset instead of a bare
first-class workgroup pointer expression. The omitted default `twiddles =
nullptr` argument is carried into the CrossGL intermediate before target
lowering. Because the materialized `use_twiddle_table = false` call chain makes
every twiddle dereference unreachable, DirectX removes that unobserved resource
parameter through the forwarding chain rather than inventing a backing buffer.
A null pointer that can be observed or dereferenced still fails closed.

The current source emits a 146,763-byte HLSL artifact with SHA-256
`3bc42b2dd3bf128bcbe1fd202763f3434d64df6e60b53da4b6e754ceff0f6e7a`
and zero project diagnostics. All 20 native-16 ``power`` shift counts are
explicitly promoted to ``int`` before HLSL shifting, matching Metal/C++ integer
promotion rather than retaining minimum-precision count semantics. Windows CI
compiles it with DXC using `cs_6_2`,
`-enable-16bit-types`, and warnings as errors, then packages and dispatches it
through Direct3D 12 WARP. The same index-1 complex impulse workload, reflected
resource layouts, physical workgroup count, and `2e-4` output tolerances are
required for this exact current checkout. This lowering does not permit
arbitrary first-class workgroup pointers: roots without a concrete shared-array
identity or extent still fail closed, and
[#1518](https://github.com/CrossGL/crosstl/issues/1518) continues to track the
broader pointer-offset contract. The historical proof remains separately
recorded under its own commit provenance; it is no longer being used as a
substitute for current-corpus evidence.

The current FFT source now also emits an 82,045-byte GLSL artifact with
SHA-256
`a1ab0c346d9143e6749e391fb971aeaed71bd84e15fedaf7a7e92808a56449bb`
and 84 source-remap mappings. OpenGL keeps the 21 reachable function constants
as 21 deferred specialization constants and uses the same `[1, 1, 64]`
workgroup contract. A fifth current-source index assertion bounds
`batch_idx + index + r`. Generic specialization lowering now follows a
statically null storage pointer through direct forwarding calls and omits it
only when every reachable use is dead; observable null uses still fail closed.
The GLSL generator also decodes materialized Metal `vec<T,N>` constructor
names and emits every collision-safe overloaded resource-specialization body.
The generated module compiles with `glslangValidator`, passes `spirv-val`, and
has 19 control barriers and no group-nonuniform instructions.

Host reflection now records each runtime `vec2` SSBO element as `float32x2`
with 8-byte size, stride, and alignment under `std430`; the two scalar argument
blocks retain their 16-byte `std140` block identities. The resulting four-
resource native ABI has no blocked variants and publishes a ready deferred
SPIR-V request. Linux CI specializes and dispatches that request through a
surfaceless Mesa llvmpipe OpenGL context. For the same index-1 complex impulse,
all 256 complex outputs match the analytical forward DFT within the required
`2e-4` tolerances; the container probe observed maximum absolute error
`9.264554161336758e-08`, verified the interface, and published the compilation
cache. This is one exact current-pinned workload and does not establish full
FFT, MLX host-runtime, or backend parity.

A dedicated project replay now translates the complete pinned `fft.metal`
source to one standalone OpenGL compute shader with a 4,096-specialization
limit and a 2,097,152-item materialization work budget. The report records 99
unique reachable template specializations, 22 function constants, no unsupported
materializations, and no project diagnostics. Four unsigned host-index
assertions and five entry-point-scoped workgroup access assertions make the
required dispatch preconditions explicit for memory extents from 256 through
4,096 elements. The Linux proof compiles the emitted GLSL to OpenGL SPIR-V 1.3
with `glslangValidator` and validates the binary with `spirv-val`. This proves
artifact construction and native toolchain acceptance; OpenGL runtime dispatch
and numerical parity with MLX's Metal backend remain outside this check.

The pinned `gemv.metal` DirectX compiler frontier verifies source SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f` and
source size 5,383 bytes before translation. The project report must contain one
clean translated artifact, all 226 materializations with no unsupported
records, no unresolved materialization residue, and no standalone pure
value-discard statements such as `lid;`. Its generated SHA-256 and byte count
are checked against the emitted HLSL. Project configuration sets
`[project.workgroup_size_rules]` for `gemv.metal` to `[32, "BN", "BM"]`; the
report must retain the normalized `["32", "BN", "BM"]` rule. Exactly 224 of the
226 materializations must be host-named; the other two are signature-selected
`elem_to_loc<uint>` and `elem_to_loc<uint32_t>` helpers. Exactly 224 report
execution entries must join the host-named records by
`(hostName, materializedName)` identity. The artifact must expose the exact
target set `CSMain`, `CSMain_2`, ..., `CSMain_224`, independently of report or
materialization list order.

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
[#1786](https://github.com/CrossGL/crosstl/issues/1786) established the exact
wave-size specialization contract. This existing GEMV library frontier does not
apply that contract, so library compilation proves that DXC accepts and
code-generates every exported function but does not establish wave semantics,
runtime execution, numerical parity, or whole-kernel semantic validity.

The pinned `gemv.metal` OpenGL gate uses the same 4,096-specialization limit and
2,097,152-item materialization work budget. It requires SHA-256
`c34db77e61c1fea01f7f5d319a0bec1029a253e54d66bbce9009f32fe828ce9f`, one
source unit, all 226 specializations materialized without unsupported records,
and all 224 entry-scoped GLSL artifacts emitted without diagnostics. Every
artifact must retain its source entry identity, generated hash, source map,
source remap, resolved workgroup size, and exact subgroup-width contract.

The project configuration records `[32, "BN", "BM"]` as the workgroup-size
rule and 32 as the subgroup width. Generated shaders require
`GL_KHR_shader_subgroup_basic` and return before dispatch work when
`gl_SubgroupSize` differs from 32; the host must also query
`GL_SUBGROUP_SIZE_KHR` and reject a mismatch before dispatch. Five source-scoped
index-range assertions record the host preconditions that permit MLX's 64-bit
batch and gathered-matrix indices to narrow to OpenGL's 32-bit index domain.
These are explicit runtime preconditions, not inferred ranges, and this harness
does not enforce them at runtime.

Linux CI compiles all 224 artifacts for OpenGL SPIR-V 1.3 with
`glslangValidator` and validates every resulting SPIR-V 1.3 module with
`spirv-val`. [#1894](https://github.com/CrossGL/crosstl/issues/1894) tracks a
semantics-preserving fallback for devices without the required native subgroup
width. The gate proves complete translation and compiler acceptance under the
recorded project contracts; it does not establish runtime execution, host
integration, or numerical parity.

The checked
[`contracts/gemv.native-loader.dispatch.json`](contracts/gemv.native-loader.dispatch.json)
contract selects current-pinned
`gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0` from the newer 6,981-byte
`gemv.metal` source with SHA-256
`0bd8bde0c867a17c345a3651f9f0a6c2909e0c74e76ea2a08f373fe4dcafaeda`.
The host-derived `gemv_axbpy` branch covers one contiguous float32 vector-matrix
product with `M=1`, `N=32`, `K=32`, no gathered or non-contiguous batch, and no
axpby bias path, matching
`python/tests/test_blas.py::TestBlas::test_matrix_vector`. It fixes parameters
`BM=1`, `BN=2`, `SM=8`, `SN=4`, `TM=4`, and `TN=4`, workgroup
`[32, 2, 1]`, subgroup width 32, and dispatch `[1, 1, 1]`. The normalized
contract identity is
`sha256:6b3bb18d130159f13874f06668b536fe4b9270ffbb2a1f44b6d9aac257aba7e4`;
its single variant and artifact identities are
`sha256:acaba2ec4813a364b06d95a5136bda80351797591a4d9f0b3d195f85da287fe3`
and
`sha256:34eab189b10cc699f06f4cbed04faae41a2658a2a3665a6866ed987f5946949a`.

Entry-scoped translation materializes only the selected GEMV and
`elem_to_loc_uint`, with no unsupported record or project diagnostic. The
8,188-byte HLSL has SHA-256
`f300bbea75b2ed9e47c29313a56f882ed848cbb93858f1347fbc97a60e167223`,
retains `[numthreads(32, 2, 1)]` and `[WaveSize(32)]`, and passes official DXC
1.9.2602.24 under `cs_6_6`, `-enable-16bit-types`, and warnings as errors.
Direct3D does not guarantee that a multidimensional workgroup's flattened
`SV_GroupIndex` values are contiguous within each physical wave, so this entry
explicitly enables target-scoped 32-lane software subgroups. Logical subgroup
and lane IDs are `SV_GroupIndex / 32` and `SV_GroupIndex % 32`; a 64-float
`groupshared` array carries each shuffle, with two
`GroupMemoryBarrierWithGroupSync` calls. The source lane is validated before
addition, preventing unsigned-delta wrap, and an out-of-range shuffle returns
the calling invocation's value. The artifact contains no `WaveReadLaneAt`,
`WaveGetLaneIndex`, or physical-wave atomic allocator. `[WaveSize(32)]` remains
a source/reflection contract, not a dependency on physical lane topology.

The replaced 8,410-byte physical-wave artifact is retained as rejected
diagnostic evidence under SHA-256
`f8f1107d0de251fd300c7a16ce6638796bd08dd2eadd8f7959e37c78d0aa170d`.
Windows workflow run 33268998061, job 99143984804 mismatched all 32 outputs:
its reduction substituted logical lanes 5 through 8 with physical lanes 21
through 24, with maximum absolute error 1.90625. That exact signature rules out
a tolerance adjustment or merely guarding invalid high-lane reads.

The OpenGL target needs only the selected matrix-index assertion
`uint64(bm + tm) * marix_ld + out_col + tn` in the unsigned 32-bit range. Its
7,705-byte GLSL has SHA-256
`f5ef8900ee65d63a6df2818ef111f56b4f269c6366c82d82a9d97c967042f562`
and partitions the 64-thread workgroup into two logical 32-lane subgroups with
a 64-float shared shuffle scratch array.

Both target-specific software-subgroup analyses recognize either `value > 0`
or the integral-equivalent `value >= 1` as a canonical positive-to-zero
halving loop when paired with `/= 2` or `>>= 1`. That admits the source's
`sm >= 1; sm >>= 1` segmented shuffle reduction. DirectX additionally requires
one bounded compute entry, a concrete width-compatible workgroup, explicit
calling-invocation fallback, a supported scalar shuffle, an unambiguous helper
call graph, logical invocation identity, and statically uniform control flow;
violations fail before artifact emission. OpenGL retains rejection for wider
bounds, mutation, nontermination, escaping control flow, indirect calls, and
nested helper calls. `glslangValidator` and `spirv-val` accept the emitted
OpenGL module; SPIR-V contains three control barriers and no group-nonuniform
instruction or hardware-subgroup extension.

Both runtime packages expose the same 15 logical resources: matrix, vector,
bias placeholder, writable output, batch shape, three signed 64-bit batch
strides, and seven scalar argument blocks. A deterministic binary-fraction
workload uses `vector[row] = (row + 1) / 32` and
`matrix[row,column] = (row + column + 2) / 64`; output column `c` must equal
`5.5859375 + 0.2578125 * (c + 1)` for all 32 columns at `1e-5` absolute and
relative tolerance. Linux arm64 Mesa llvmpipe executes and reads back this
software-subgroup workload in required mode, and Windows CI requires the same
request through Direct3D 12 WARP.

This bounded proof does not replace the historical 224-entry aggregate gates:
it adds numerical execution for one host-valid entry from the materially newer
current corpus. Gather, wide, batched, axpby, and the remaining host-named GEMV
entries, MLX host redirection, the full MLX test suite, and selected-entry Metal
compiler validation remain outside the claim. The separate native Metal
aggregate baseline continues to cover the existing round-trip boundary.

The checked
[`contracts/fp_quantized.native-loader.dispatch.json`](contracts/fp_quantized.native-loader.dispatch.json)
contract selects current-pinned
`mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false` from the 9,700-byte
`fp_quantized.metal` source with SHA-256
`ef4ba099710a63a0b5d27d3e5ce69a8528bee8f1757805aa606c8d8e43de18d4`.
It records the exact host branch from `quantized.cpp` and implementation in
`fp_quantized.h`: float32 MXFP4, group size 32, four payload bits, row-contiguous
layout, and no global scale. The host formula assigns one value per thread,
selects workgroup `[32, 1, 1]`, and dispatches one workgroup. The normalized
contract identity is
`sha256:5256e32b364ac303a6873f28b5ac3e9a1a811ac5c38bc41a977bce9191a025ed`;
its single variant and artifact identities are
`sha256:ebd6ab3f40f5839764f592943180ba64f11a66f10a5561b9468019032c04df8a`
and
`sha256:bde1bfa31c116a52a1dc3b6e546dfa2ee43dc968719393ba04f629b4e2d95319`.

Entry-scoped translation materializes only the selected specialization with
`T=float`, `group_size=32`, `bits=4`, and `has_global_scale=false`. The
9,123-byte HLSL has SHA-256
`3fe38e171ba8c8ea1adfc8efad20b242ca02dd05e1a5a53a9b9d1e18459d8c7d`,
retains `[numthreads(32, 1, 1)]` and `[WaveSize(32)]`, and passes DXC under
`cs_6_6`, `-enable-16bit-types`, and warnings as errors. Its compiled DXIL is
4,716 bytes. This artifact explicitly enables the DirectX-only
`project.source_options.metal.target_options.directx.widen_native_float16`
mode. Source `as_type<float16_t>(uint16_t)` reconstructs its exact payload as
float32 with integer IEEE-754 masks, and logical `float16_t` locals, function
parameters, and returns stay widened through the selected arithmetic path.
Optimized DXIL contains `uitofp i32` and `fmul float`, with no
`LegacyF16ToF32`, `half`, `fptrunc`, or `fpext`. Default DirectX native
binary16 lowering remains exact `asfloat16`/`asint16`/`asuint16` and is
unchanged unless this source-scoped option is enabled.

Four rejected Windows artifacts establish why every native-half boundary must
be absent. The 7,809-byte HLSL under SHA-256
`3591e38d20a612b4061fe3154ef0ea3deb035283294fbd27376ef90627569361`
produced numeric `uitofp i16 ... to half`; workflow run 33271117475, job
99149649480 collapsed all 28 nonzero values to signed zero on WARP. Exact
`bitcast i16 ... to half` produced same-size HLSL under SHA-256
`4e8044758d65b6b2c189092ce56fff3c5ba7948221883de490c1a4b9c5563352`,
but run 33272842347, job 99154326814 still consumed the intentionally
constructed binary16 subnormal in `fmul half` and produced the same 28 signed
zeros. Moving the multiply to float32 produced 7,909-byte HLSL under SHA-256
`938ca6fac1c47ea633453836b5d76833c294853bb92d6a410a2c4772dd7fa627`,
but `dx.op.legacyF16ToF32` yielded the identical result in run 33274360343,
job 99158370210. Integer decoding then produced 9,240-byte HLSL under SHA-256
`936088a24a6b575e50dc97e16a4c0dca63a76200ddd94d5211e4bf312fec1625`,
but its remaining `fptrunc float to half`, half sign operation, and
`fpext half to float` still returned the same 28 signed zeros in run
33275550062, job 99161501105. Removing every half instruction produced the
9,118-byte artifact under SHA-256
`7afdc612f9091ae47abca8c4fd9d2171e8ea42c6539e02a40bbad2de7d1a7c6a`,
but run 33277494856, job 99166677942 still returned the same signed zeros.
Its DXIL exposed the actual remaining defect: Metal/C++ scalar integer promotion
was missing, so `(uint16_t(bits) << 23)` became a 16-bit shift reduced to 7 and
masked with `0xffff`. The corrected HLSL emits
`int(uint16_t(bits)) << 23`; DXIL now shifts the 32-bit value by 23 before
`asfloat`, while retaining zero native-half instructions.

The 9,571-byte GLSL has SHA-256
`cbbe989c40317c04ffe915f1f314f55db8896edfd38f04ad4b8882be53b2a4da`,
uses one explicit 32-lane software subgroup for `WaveActiveMax(float)`, and
passes `glslangValidator` and `spirv-val`. Its 10,488-byte SPIR-V has three
control barriers, no group-nonuniform instruction, and local size
`[32, 1, 1]`. Because GLSL widens source binary16 values to float32, the same
source bitcast preserves the low 16-bit payload through `unpackHalf2x16` rather
than incorrectly reinterpreting the widened integer as a 32-bit float; the
inverse form uses `packHalf2x16` and exact low-bit extraction.

The scale conversion invokes the source `fp8_e8m0(float)` constructor factory
before the selected sibling float conversion operator; aggregate field
initialization would encode the scale incorrectly. Qualified `metal::round`
lowers to the portable floating intrinsic. OpenGL lowers `metal::isfinite` to
a single-evaluation IEEE-754 float32 exponent-mask test and `signbit` to the
exact sign bit. The private `fp8_e4m3` scalar view is admitted only as a
read-only, exact one-member-layout projection. Unresolved constructor branches,
unsupported predicate types, writes, and receiver mutation remain fail-closed.

The reflected data ABI is float32 input at binding 0, the retained but
statically unread `global_scale` input at binding 1, and float32 read-write
output at binding 2. DirectX additionally reflects generated dispatch input at
`b0`; HLSL register namespaces make this legal beside input `t0`, while
`global_scale` uses `t1` and output uses `u2`. The real no-global-scale host
branch omits buffer 1. The generic reflected-resource loader instead supplies
one inert allocation because it cannot omit a declared resource, and the test
checks that the selected specialization contains no read of it.

The numerical request contains 32 exact FP4 E2M1 values from -6 through 6.
Its maximum absolute value and scale divisor are both exactly 6, so the encoded
MX scale is exactly 1 and quantize/dequantize must return every input bit-exactly
at zero absolute and relative tolerance. Windows CI requires that request
through Direct3D 12 WARP; Linux CI requires the software-subgroup artifact
through Mesa headless EGL. Both paths build the runtime artifact manifest,
package, loader manifest, reflected ABI descriptor, and one-workgroup dispatch
request before native execution.

This is one bounded float32 MXFP4/no-global-scale workload. It does not cover
the remaining `fp_quantized` entries, global-scale variants, other group sizes,
bit widths or dtypes, MLX host redirection, selected-entry Metal compilation,
or the full MLX test suite. The separate native Metal aggregate baseline remains
the round-trip boundary.

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
`arg_reduce.metal` still materializes all 24 host-named entries. The historical
aggregate run emits Vulkan but continues to fail closed for DirectX and OpenGL
with `project.translate.workgroup-size-entry-ambiguous`, because one aggregate
artifact cannot infer the runtime-selected axis and pipeline limits. That
aggregate result no longer describes all available arg-reduce coverage.

The checked
[`contracts/arg_reduce.native-loader.dispatch.json`](contracts/arg_reduce.native-loader.dispatch.json)
contract now selects current-pinned `argmin_float32` and `argmax_float32` for
two axis-32 rows. It applies the host formula
`roundUp(min(ceilDiv(axisSize, 4), maxThreadsPerWorkgroup), simdWidth)`, fixes a
wave width of 32, emits `[32, 1, 1]`, and dispatches `[1, 2, 1]` workgroups.
Signature-aware helper materialization selects the scalar
`elem_to_loc<int64_t>` overload rather than the `uint3` overload. The HLSL
artifacts are 6,655 and 6,657 bytes with SHA-256
`e3f7392023bbb6457eb03398a766bdaa128ed709d66ce814c7209cd13de7e896`
and `ef67c5d24ae7c7492a6676a35e0604800c1d18e4113c411fffaa2070090a92c3`;
official DXC 1.9.2602.24 accepts both under `cs_6_6`,
`-enable-16bit-types`, and warnings as errors. This proof explicitly sets
`project.source_options.metal.target_options.directx.relative_wave_shuffle_out_of_range`
to `"self"`: a relative shuffle whose source lane is outside the wave retains
the calling lane's value. Generated helpers select either the valid relative
lane or the calling lane before every unconditional `WaveReadLaneAt`, so the
second reduction cannot consume undefined high-lane state. The default DirectX
policy remains `"undefined"`, preserving existing artifacts unless a project
opts in.

The explicit OpenGL software-subgroup artifacts are 7,581 and 7,587 bytes with
SHA-256
`b74534a5120665ad07755141af2a73702cb5ea504a0526b92306eabedfed4765`
and `d90e758132832490b7f356c6750d4deb2bdb3341f053a921228a9f24ce8d27d8`.
They admit direct shuffle-helper calls only inside a canonical workgroup-uniform
halving loop (`offset > 0` with `/= constant >= 2` or `>>= constant >= 1`),
while lane-varying, nonterminating, escaping, mutated, indirect, and nested
forms remain rejected. Both modules pass `glslangValidator` and `spirv-val`,
contain five control barriers, and contain no group-nonuniform SPIR-V
instruction.

The native ABI reflects float32 input, uint32 output, int32 shape, int64 stride,
and uint64 size resources exactly: nine DirectX bindings include the generated
`CrossGLDispatchInfo`, while OpenGL has eight bindings. Linux arm64 llvmpipe
executed deterministic rows with repeated extrema and read back argmin indices
`[5, 7]` and argmax indices `[3, 2]`, proving lowest-index tie behavior.
Windows CI requires the same two workloads through Direct3D 12 WARP and Linux
CI requires them through surfaceless Mesa EGL. Other axes, dtypes, and the
remaining 22 host-named entries, MLX host redirection, and the full MLX test
suite remain outside this bounded claim. Entry-scoped Metal output still fails
explicitly at workgroup specialization with
`project.translate.workgroup-size-rule-unsupported-target`; the native macOS
aggregate baseline is separate and this proof does not claim an entry-scoped
Metal round trip. Project dispatch import was completed under
[#1793](https://github.com/CrossGL/crosstl/issues/1793), and remaining broad
entry packaging stays tracked in
[#1523](https://github.com/CrossGL/crosstl/issues/1523).

The checked
[`contracts/scaled_dot_product_attention.native-loader.dispatch.json`](contracts/scaled_dot_product_attention.native-loader.dispatch.json)
contract now selects current-pinned `sdpa_vector_float_64_64` for the bounded
one-pass vector path: batch 1, one query and KV head, query length 1, key length
4, query/value dimensions 64, scale 0.125, and no mask, causal mode, or sinks.
The host-derived contract fixes workgroup `[1024, 1, 1]`, subgroup width 32,
and one dispatched workgroup. Function constant IDs 20 through 25
(`has_mask`, `query_transposed`, `do_causal`, `bool_mask`, `float_mask`, and
`has_sinks`) are all false; ID 26 belongs to the two-pass path and is
intentionally absent. The dispatch content identity is
`97e5ebb69af8da3a0082776015787456f23b8bfdb0cff757f5364db2cfef8d2c`,
with variant ID
`sha256:8b2abb9f7179e051530697fb8d1956d0ff03a324e7acaa5fcdf4f4dd9f1befbb`
and artifact ID
`sha256:dd0138695bd82e1f8ea49bd667052b484420ee96cb2849c6eed20ba5eae39a89`.

The 8,721-byte HLSL artifact has SHA-256
`003c8b9e85bad7363bae2e3d80380d979cbe0b8988d0d98751131c3acfbff6b6`.
Official DXC 1.9.2602.24 accepts `CSMain` under `cs_6_6`,
`-enable-16bit-types`, and warnings as errors, producing 9,000 bytes of DXIL.
Its 32 physical waves receive unique workgroup-synchronized subgroup IDs; no
lane-varying `SV_GroupIndex / WaveGetLaneCount()` derivation remains. The
12,089-byte explicit software-subgroup GLSL artifact has SHA-256
`9b7cb7dc9a76b9fb93c30fd93d13ad639f5493f60fd97b965514db0fe6b4840b`.
It partitions 1,024 invocations into 32 logical subgroups and synchronizes the
source subgroup-ID-strided runtime loop across every workgroup round. Inactive
subgroups contribute typed collective identities, so all generated barriers
remain uniform. `glslangValidator` and `spirv-val` accept the resulting
OpenGL/SPIR-V 1.3 module; disassembly contains exactly nine
`OpControlBarrier` instructions, six `OpSpecConstantFalse` declarations,
local size `1024 1 1`, and no `OpGroupNonUniform` instruction.

Both targets package complete native-loader requests. DirectX has 19 reflected
resources including generated `CrossGLDispatchInfo`; OpenGL has 18. Optional
`bmask`, `fmask`, mask-stride, and sinks bindings receive harmless placeholders.
The stored Boolean mask ABI is physically uint32 in both HLSL and GLSL block
storage. DirectX concretizes the six false constants. OpenGL retains them for a
verified deferred GLSL-to-SPIR-V specialization request and publishes the JSON
variant registry while explicitly omitting the unsupported native registry
header.

Windows CI requires the 64-value output to execute through Direct3D 12 WARP.
Linux CI binds PyOpenGL to surfaceless EGL, forces llvmpipe, performs deferred
SPIR-V specialization, dispatches through the OpenGL native loader, and compares
all values with a stable CPU scaled-attention reference at `2e-4` absolute and
relative tolerance. The local Mesa run had maximum absolute error
`4.082320426146424e-08` and maximum relative error
`4.2163276126605175e-06`. This is bounded evidence for one float32 one-pass
workload only. Masked, causal, sinks, two-pass and full-attention paths, other
dimensions and dtypes, the remaining host-named entries, MLX host redirection,
and the full MLX test suite remain outside the claim. The separate native Metal
baseline does not make this selected native-loader proof a Metal round trip.
The historical 42-entry aggregate DirectX/OpenGL result remains fail-closed
because it does not consume this entry-scoped contract; it no longer means that
no bounded attention dispatch, package, or numerical runtime proof exists.

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
binary. Its reduced native specialization check establishes typed dispatch and
deterministic readback only, not numerical parity for those full aggregate
modules. The bounded `sdpa_vector_float_64_64` proof above is separate and does
compare every selected output with a CPU attention reference. For the pinned
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

A separate native-loader check selects `v_copyfloat32float32` from the current
corpus at commit `846d176227a0ac13d2667e58d2bb68b322109ab0`. It materializes
`copy_v<float, float, 1>` and the reachable `cast_to<float, float>` helper while
pruning 62,424 unreachable candidate pairs. It packages one HLSL artifact and
one GLSL artifact with exact reflected scalar buffer layouts, dispatches eight
nonuniform float32 values through Direct3D 12 WARP on Windows and a headless
OpenGL software context on Linux, and requires exact readback on both targets.
The generated artifact hashes, byte counts, binding layouts, and 1-by-1-by-1
workgroup sizes are fixed in the test. This is bounded execution evidence for
one scalar copy entry. It does not cover the complex or bfloat16 entries,
redirect the MLX host runtime, or run the MLX test suite. Aggregate input layout
for `s_copycomplex64float32` remains part of the physical-resource contract
tracked by [#1543](https://github.com/CrossGL/crosstl/issues/1543).

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

The project-porting harness consumes this contract against the unchanged pinned
`layer_norm.metal` source and requires two deterministic DirectX artifacts. It
checks source and contract identities, entry-scoped execution provenance,
host-derived workgroup sizes, function-constant values, exact subgroup-width
enforcement, generated hashes, and artifact paths. Windows CI compiles both
artifacts with DXC `cs_6_6` and warnings as errors. This extends the bounded
proof into the repository-level project report; it does not add runtime
execution or numerical parity for those historical axis-size-4099 forward and
axis-size-8192 VJP records.

A separate current-corpus proof selects the forward `layer_normfloat32` entry at
commit `846d176227a0ac13d2667e58d2bb68b322109ab0` through
[`contracts/layer_norm.native-loader.dispatch.json`](contracts/layer_norm.native-loader.dispatch.json).
It records the exact host formula
`32 * ceil_div(ceil_div(axis_size, 8), 32)`, the upstream
`python/tests/test_fast.py::test_layer_norm` provenance, axis size 32, two rows,
a `[32, 1, 1]` workgroup, subgroup width 32, two dispatch workgroups, and no
function constants. An explicit source/entry-qualified workgroup-access
precondition bounds specialized helper views to indices 0 through 31; it is a
stated host portability contract, not an inferred or runtime-enforced bound.
Materialization emits the forward entry plus `initialize_buffer<1>` and
`threadgroup_sum<1>` from six reachable specializations while pruning 194
candidates.

The generated HLSL artifact is 5,216 bytes with SHA-256
`7e790d4e665c72025e46c7c038aba2bec57ba6f65e209178eae5160c0c7ea8e9`.
It retains `[WaveSize(32)]`, two `WaveActiveSum` calls, and compiles as
`cs_6_6` with `-enable-16bit-types`; Windows CI requires Direct3D 12 WARP
execution. The generated GLSL artifact is 5,914 bytes with SHA-256
`f86f83b6835b7d4b07ece9f153df883300f7a131bbcec5d084bf29084c1bf51a`.
Its explicit target-scoped software path admits a subgroup helper only when the
helper has one stable non-overloaded source identity and every call is direct,
unconditional, top-level, and made from the sole compute entry. Conditional,
nested, indirect, ambiguous, or potentially divergent helper use rejects
translation. Resource-specialized helper lookup may reuse a prevalidated
workgroup proof only when its base key is identical and its intervals cover the
requested narrower proof; incompatible or narrower assumptions remain
rejected.

The software GLSL contains the shared-memory sum helper and six barriers, but no
KHR subgroup extension, `gl_Subgroup*` use, `subgroupAdd`, or SPIR-V
`OpGroupNonUniform` instruction. `glslangValidator` and `spirv-val` accept it;
its disassembly contains six `OpControlBarrier` instructions. The native-loader
package reflects eight ready resources on both targets: float32 input, weight,
bias, and output arrays, plus 16-byte float32 epsilon and uint32 axis-size,
weight-stride, and bias-stride blocks. DirectX uses structured/constant-buffer
layouts and OpenGL uses std430/std140 layouts.

The deterministic workload uses two 32-element rows, 32 weights, 32 biases,
epsilon `1e-5`, and unit weight/bias strides. It compares 64 outputs with
`((x - mean) / sqrt(mean((x - mean)^2) + epsilon)) * weight + bias` at `5e-5`
absolute and relative tolerance. A local Linux arm64 llvmpipe EGL dispatch
passed with maximum absolute error below `8.88e-8`; Linux CI requires the same
Mesa numerical readback. This is bounded execution evidence for one float32
forward workload only. It does not redirect MLX host execution, run the MLX
test suite, or cover VJP, float16, bfloat16, looped entries, other axis sizes,
or the historical wider dispatch records above.

A sibling current-corpus proof selects `vjp_layer_normfloat32` at axis size 32
through
[`contracts/layer_norm_vjp.native-loader.dispatch.json`](contracts/layer_norm_vjp.native-loader.dispatch.json).
It fixes one float32 row, `has_w=true` through function constant ID `20`, a
`[32, 1, 1]` workgroup and 32-lane subgroup, one dispatch workgroup, and the
explicit helper-access interval 0 through 95 required by three SIMD-sized
threadgroup slices. The pinned
`python/tests/test_fast.py::test_layer_norm_grad` case supplies the upstream
shape and weighted-gradient provenance. Materialization selects
`vjp_layer_norm_single_row<float, 8>`, `initialize_buffer<3>`,
`threadgroup_sum<3>`, and `threadgroup_sum<1>` from seven reachable
specializations while pruning 194 candidates.

The generated HLSL is 7,504 bytes with SHA-256
`6d4a3281d038309c8c294952411acfeb773f6ee8ddd8d73935cb3f3c4ce93a61`.
It concretizes `has_w=true`, retains `[WaveSize(32)]` and four
`WaveActiveSum` calls, compiles as `cs_6_6` with `-enable-16bit-types`, and
must execute numerically through Direct3D 12 WARP on Windows CI. The generated
software-subgroup GLSL is 8,291 bytes with SHA-256
`9e6c4e6201e1c78e981a346275b849c37e6c8d834e7509d662f7aec5782980fa`.
It retains deferred OpenGL specialization constant `20`, emits eight control
barriers with no hardware-subgroup extension or SPIR-V group-nonuniform
instruction, and passes `glslangValidator` plus `spirv-val`.

The OpenGL ABI package keeps its exact JSON runtime variant registry ready and
actionable. Its generated native C++ registry header is deliberately marked
unavailable with reason `specialization-requires-deferred-compilation`, because
the GLSL source still requires specialization. The runtime derives a bounded
deferred-compilation request, compiles GLSL to SPIR-V, applies `has_w=true`
through the OpenGL SPIR-V specialization API, then dispatches the resulting
module. This is not an unavailable or blocked runtime variant.

The eight-resource native-loader ABI contains float32 `x`, `w`, `g`, `gx`, and
per-row `gw` buffers plus float32 epsilon, uint32 axis-size, and uint32
weight-stride scalar blocks. The deterministic request compares all 32 `gx`
and 32 `gw` values at `7e-5` absolute and relative tolerance. A local Linux
arm64 llvmpipe EGL run passed with maximum absolute errors below `4.82e-8` for
`gx` and `5.44e-8` for `gw`; Linux CI requires the same deferred SPIR-V Mesa
execution.

This proof is intentionally one row: the one-row boundary makes the per-row
`gw` temporary equal to the host's final reduced weight gradient. It does not
include the separate bias-gradient reduction, multi-row weight reduction,
`has_w=false`, float16 or bfloat16, looped entries, other axis sizes, MLX host
redirection, or the full MLX test suite.

Validate the fixture schema, provenance, deterministic identities, and bounded
evaluation with:

```bash
.venv/bin/python -m pytest -q -n auto \
  tests/test_mlx_dispatch_contract_fixture.py
```

The checked-in
[`contracts/logsumexp.dispatch.json`](contracts/logsumexp.dispatch.json)
fixture captures two float32 block-reduction workloads from the pinned MLX
operation and gradient tests. The records use axis sizes 32 and 1025 to exercise
distinct results of the pinned host formula
`32 * ceil_div(ceil_div(axis_size, 4), 32)`: workgroup sizes `[32, 1, 1]` and
`[288, 1, 1]`. Both dispatch one output row, require a 32-lane subgroup, and
select `block_logsumexp_float32`. The looped path for axis sizes above 4096 and
the float16 and bfloat16 entries remain outside this bounded contract.

The companion
[`contracts/logsumexp.native-loader.dispatch.json`](contracts/logsumexp.native-loader.dispatch.json)
pins the same bounded workloads to current corpus commit
`846d176227a0ac13d2667e58d2bb68b322109ab0`. The upstream kernel, host dispatch
formula, and referenced workload coverage are unchanged between the two
revisions, so the two evaluated variant sets are identical; only revision
provenance differs. Current-corpus Direct3D execution, OpenGL toolchain, and
bounded OpenGL software-runtime tests use this companion contract. The
historical contract remains attached to the recorded 40-unit frontier.

The repository harness evaluates both records against the unchanged pinned
`logsumexp.metal` source and emits one standalone DirectX artifact per workload.
It verifies source, contract, dispatch, template-materialization, and artifact
identities; checks the generated max and sum wave reductions, group barriers,
exponential, logarithm, output store, workgroup size, and `[WaveSize(32)]`; and
requires official DXC `cs_6_6` compilation with warnings as errors on Windows.
This establishes translation and native compiler acceptance for the two listed
test-derived dispatches.

Both revision-specific contracts produce the same two dispatch variants as
standalone OpenGL artifacts.
Each generated GLSL file requires `GL_KHR_shader_subgroup_basic`, declares
`CROSSTL_REQUIRED_SUBGROUP_WIDTH` as 32, and checks `gl_SubgroupSize` before any
translated kernel work. The portability report records the matching
`GL_KHR_shader_subgroup` host requirement and `GL_SUBGROUP_SIZE_KHR` query, and
the runtime artifact manifest retains both the local size and exact subgroup
width. Linux CI requires `glslangValidator` and `spirv-val` to accept both
artifacts. The built-in Python runtime and generated C++ adapter reject a
reported width other than 32 before compiling the shader or allocating its
resources.

The default OpenGL path remains the guarded hardware-subgroup path described
above. A separately selected current-corpus axis-size-32 artifact opts into
``project.source_options.metal.target_options.opengl.software_subgroup_width =
32``. Because its complete workgroup is one 32-lane subgroup, CrossTL can prove
that ``simdgroup_index_in_threadgroup`` is zero and that
``thread_index_in_simdgroup`` is ``gl_LocalInvocationIndex``. It lowers the two
scalar ``WaveActiveMax(float)`` and ``WaveActiveSum(float)`` reductions through
shared-memory trees while continuing to reject direct raw subgroup builtins,
helper-contained operations, unsupported payloads, and unproven divergent
control flow. The default hardware artifacts and host width preflight are
unchanged.

The software artifact is 4,676 bytes with SHA-256
``813762d4535fdd693ca0a48c3c3f5dc79f6cc298050faae6e180d3cc9f1d60e5``.
It contains ``CROSSTL_SOFTWARE_SUBGROUP_WIDTH``, no KHR subgroup extension or
``gl_Subgroup*`` use, and no hardware subgroup execution metadata.
``glslangValidator`` and ``spirv-val`` accept its OpenGL/SPIR-V 1.3 module; the
SPIR-V contains ten control-barrier instructions and no group-nonuniform
instruction. Runtime reflection packages ``in_Buffer`` and ``out_Buffer`` as
std430 float32 resources at bindings 0 and 1, plus the 16-byte std140
axis-size block at binding 2, with one ready native load unit.

Linux CI requires a surfaceless Mesa EGL dispatch of that software artifact.
The workload binds the 32 values ``(index - 16) / 8``, sets axis size 32,
dispatches one 32-thread workgroup, and compares the single readback with the
stable CPU LogSumExp reference ``3.9978051379373145`` at ``5e-5`` absolute and
relative tolerance. The Windows native-loader test exercises the same bounded
workload through the guarded HLSL artifact and Direct3D 12 WARP. This is exact
translation, packaging, and numerical execution coverage for one finite
current-corpus workload; it does not redirect the MLX runtime or run the MLX
test suite.

The axis-size-1025 record still produces a ``[288, 1, 1]`` artifact containing
nine logical 32-lane subgroups, but it remains outside this LogSumExp software
runtime package. The compiler now admits structurally constrained typed masked
sum, minimum, and maximum reductions, as exercised by the current-corpus
Softmax proof below; this LogSumExp workload has not been packaged or
numerically dispatched under that mode here. Its checked artifact therefore
retains the compatible width-32 hardware contract. Looped reductions, other
dtypes and shapes, and full-runtime parity remain tracked by
[#1894](https://github.com/CrossGL/crosstl/issues/1894).

Validate the LogSumExp contract schema, provenance, deterministic identities,
and bounded workload set with:

```bash
.venv/bin/python -m pytest -q -n auto \
  tests/test_mlx_logsumexp_dispatch_contract_fixture.py
```

The checked-in
[`contracts/softmax.native-loader.dispatch.json`](contracts/softmax.native-loader.dispatch.json)
fixture selects current-pinned `block_softmax_float32` and records two
host-derived block workloads. Axis size 32 with two rows uses `[32, 1, 1]` and
dispatches two workgroups. Axis size 2049 with one row uses `[544, 1, 1]` and
dispatches one workgroup; 2049 is directly represented in the pinned upstream
`python/tests/test_ops.py::test_softmax` coverage. Both follow
`32 * ceilDiv(ceilDiv(axisSize, 4), 32)`, stay within the host's block-path
limit of 4096, require logical subgroup width 32, and package three scalar
resources: float32 input and output buffers plus the int32 axis-size block.

DirectX emits one guarded `CSMain` artifact for each workgroup size, retaining
`[WaveSize(32)]`. Official DXC 1.9.2602.24 accepts both under `cs_6_6` with
`-enable-16bit-types` and warnings as errors. Their SHA-256 values are
`8b5540acc90669bc8b4a75985b42ee34c9e45258c63889f703f140b1330337ee`
and `1c20679115f29d981762165f7c9e1ecd57a641ceff376b2f8d13f33520857f05`.
The 32-thread entry is provably one wave and keeps the zero-ID quotient fast
path; the 544-thread entry allocates one uniform ID per physical wave through a
workgroup-synchronized counter. The default OpenGL path remains a separate
guarded hardware-subgroup artifact.
The runtime proof opts into explicit software subgroups instead, producing a
5,585-byte axis-32 artifact with SHA-256
`f69dad597cefc34f7908799aaf0ba2eac47a0dcdd91e5f2bf3d7247172fa84b9`
and a 7,204-byte axis-2049 artifact with SHA-256
`eb195e15089f4e7bade380af55e8b7e167c4b89f80b2f25675eb71196a5468ce`.
Both pass `glslangValidator` and `spirv-val`, contain exactly 11
`OpControlBarrier` instructions, and contain no `OpGroupNonUniform` operation.

The 544-thread artifact partitions the workgroup into 17 independent logical
subgroups. Its subgroup-zero final reductions execute under a lane-dependent
condition, so CrossTL evaluates the branch prefix only for active lanes and
contributes typed identities from inactive lanes before calling the uniform
barriered helper: negative infinity for float maximum and zero for float sum.
The same fail-closed lifting supports 32-bit float, int, and uint sum, minimum,
and maximum identities. Conditional shuffle and structurally ambiguous masked
collectives remain rejected.

Windows CI requires both parameterized workloads to execute through the
Direct3D 12 WARP native loader. Linux CI requires both software artifacts to
execute through surfaceless Mesa EGL; a local Linux arm64 llvmpipe run passed
both workloads. Every output is compared with a stable float32 CPU Softmax
reference at `5e-5` absolute and relative tolerance. This is bounded evidence
for two block float32 workloads only: the looped path above axis 4096, float16
and bfloat16 variants, MLX host redirection, and the full MLX test suite remain
outside the claim. Selected entry-scoped Metal generation currently fails
explicitly with `project.translate.entry-point-target-unsupported`, so this
proof does not claim the Metal round trip.

The historical reduced-frontier aggregate can still list `softmax.metal` under
workgroup-size dispatch blockers because that aggregate run does not consume
this later current-corpus entry-scoped contract. That result describes the
aggregate pipeline coordinate; it does not mean that no bounded Softmax
translation, package, or native runtime integration exists.

The current-corpus dot-product proof selects
`dot_product_float32_it32_tg512_sg16` directly from `dot.metal`. Translation
materializes the concrete `dot_product<float, 32, 512, 16>` entry with a
512-thread workgroup and an exact subgroup width of 32. The generated artifacts
preserve the read-only `float4` storage views as four bit-preserving scalar
loads, both subgroup reductions, the 16-element workgroup reduction buffer, and
the final output store. Their portability report and runtime manifest retain
the source entry, target entry, workgroup size, subgroup requirement, and all
four reflected resource layouts. For DirectX, the 16 physical waves receive
unique, wave-uniform IDs through the synchronized allocator rather than a
lane-varying flattened-index quotient.

Windows CI compiles the HLSL artifact with DXC, packages its native-loader ABI,
and dispatches one workgroup through Direct3D 12 WARP. The bounded workload
computes the dot product of 1,024 float32 values containing `1.0` and `0.25` and
requires a readback of `256.0`. Linux CI independently preserves and compiles
the guarded 5,188-byte GLSL artifact with `glslangValidator`; its SHA-256 remains
`ef69a757339fe09897a38804c27be279a19a7db146e2e02f85f0349c59f3168d`, and it
continues to reject a device subgroup width other than 32 before translated
work.

A separate software artifact is 6,275 bytes with SHA-256
`a3c1958daa680419ce3f38559de1a6a2319a7abdac556a049632194c88223a32`.
It allocates 512 shared float elements and partitions them into sixteen
independent 32-lane logical subgroups. The first sum reduces within each
partition. For the second sum beneath `tid < 16`, inactive lanes contribute the
additive identity while all 512 invocations reach the helper barriers uniformly;
only the active lanes consume the subgroup-zero result. The artifact emits no
KHR subgroup extension, hardware `gl_Subgroup*` builtin, or group-nonuniform
SPIR-V operation. `glslangValidator` and `spirv-val` accept its SPIR-V 1.3
module, whose disassembly contains four control barriers and no group-nonuniform
instruction. Mesa llvmpipe executes the same workload through surfaceless EGL
and requires the same `256.0` readback at `1e-5` absolute and relative
tolerance.

macOS CI records that the equivalent selected Metal round trip still fails
closed at storage-backed vector pointer lowering under
[#1903](https://github.com/CrossGL/crosstl/issues/1903); it does not claim native
Metal compiler acceptance. These checks establish cross-target numerical
execution for one selected kernel workload and do not redirect MLX host
dispatch, cover the float16 or bfloat16 dot entries, or run the MLX test suite.

The current-corpus unary proofs select `v_Squarefloat32float32` and
`v_ArcCosfloat32float32` from the full include-expanded `unary.metal` source.
Each entry-scoped artifact emits exactly one
`unary_v<T=float, U=float, Op=..., N=1>` specialization. Concrete call-argument
typing keeps the out-of-line complex `ArcCos::operator()` body out of the float
artifact while preserving fail-closed handling when that complex overload is
reachable. Square retains `x * x`. ArcCos retains the source
`metal::precise::acos` contract through a portable float32 range-reduction
implementation with no-contraction qualifiers instead of relying on the target
intrinsic's unspecified accuracy. OpenGL materializes each precise helper result
through a collision-safe `precise` local, avoiding non-portable qualified
function return types while preserving SPIR-V `NoContraction`. The generated
artifacts also retain a one-thread workgroup, the source and output buffers, and
the size constant in their reflected runtime interfaces. Deterministic artifact
hashes and the pinned upstream source hash are checked before packaging.

The Square and ArcCos entries now also round-trip through Metal. Square is one
1,015-byte artifact with SHA-256
``244e34b7aa58b7abe7c3ff09f3f51f3aa283a42bf7585bf88200590767032495``;
ArcCos is one 2,742-byte artifact with SHA-256
``1247739bc0c48d11692aee81953d8a6a4071de488bfe7ea8d7b2083aa48d9b2b``.
Entry reachability retains only the selected ``struct Square`` or ``struct
ArcCos`` and its call helpers, with no unrelated unary struct, complex ArcCos
body, or illegal ``[[static]]`` member. The ArcCos artifact retains the
portable float32 range-reduction helper and brackets both helper regions with
``#pragma clang fp contract(off)`` so the source precise-math contract is not
silently weakened. Bounded Metal source reflection records each exact kernel
plus read-only buffer 0, read-write buffer 1, and read-only constant buffer 2.
Their ``[1, 1, 1]`` workgroup sizes are explicitly host-dispatch-owned because
MSL has no fixed source attribute equivalent to HLSL ``numthreads``. Both exact
artifacts compile with ``xcrun -sdk macosx metal -c`` on macOS CI.

The required family gate now covers all 877 discovered current-pinned unary
entries, adding the complete 694-entry non-scalar frontier to the earlier 183
scalar ``v_`` entries. The shape split is exact: 183 ``v_`` kernels instantiate
``unary_v`` with explicit ``N=1``; 183 ``v2_`` kernels instantiate ``unary_v2``
with the source default ``N=WorkPerThread<T>::n``; 145 ``vn_`` kernels use that
same default on ``unary_v``; 183 ``gn1_`` kernels instantiate ``unary_g`` with
``N=1`` and ``IdxT=int``; and 183 ``gn4large_`` kernels instantiate
``unary_g`` with ``N=4`` and the source-default ``IdxT=int64_t``. Together they
classify 37 operators, 20 concrete input/output type pairs, and 16 semantic
families.

Every selected run traverses Metal to CrossGL to Metal with zero project
diagnostics, emits one exact operator implementation and kernel, prunes
non-selected operator bodies, and retains only reachable empty helper tags.
The three vector shapes materialize one kernel specialization. Each gather
shape additionally materializes exactly one call-site ``elem_to_loc``
specialization, for 1,243 specializations across 877 artifacts. ``v_``, ``v2_``,
and ``vn_`` reflect input, output, and size resources; ``gn1_`` and
``gn4large_`` reflect those data buffers plus constant shape/stride buffers and
the read-only device ``ndim`` binding.

The generic round-trip support resolves and elides chained source typedefs,
maps native bfloat types and result reconstruction, materializes constrained
free operators with source-compatible overload selection, infers concrete
aggregate aliases, recognizes branch-complete returns, and preserves narrow
``as_type`` storage. FP8 decode admits only the proven read-only immediate
thread-local view of a scalar parameter as a matching single-field aggregate;
local storage, multiple or mismatched fields, escaped pointers, and every other
unimplemented pointer reinterpretation remain fail-closed. The non-scalar
increment additionally preserves ``constant`` pointer provenance after portable
``StructuredBuffer`` lowering, keeps ``const device`` references read-only, and
retains postfix ``out_idx++`` semantics. Retained source union layout metadata
is reconstructed as a native MSL ``union`` rather than an ignored attribute,
and additive shift operands retain explicit grouping for warning-fatal native
compilation.

Every source entry, shape, template, operator, exact ``T``/``U`` pair, semantic
family, SHA-256, byte count, template-default provenance, materialization count,
and host resource contract is pinned by
[`contracts/unary.metal-roundtrip.json`](contracts/unary.metal-roundtrip.json)
and linked by hash from ``expected-gaps.json``. Native macOS CI invokes
``xcrun -sdk macosx metal -Werror -c`` for every entry and requires 877
non-empty AIR outputs with no warning exemption. This closes selected-entry
translation, reflection, and native compiler coverage for every discovered unary
instantiation. It remains a compiler proof, not Metal numerical execution,
DirectX/OpenGL whole-family coverage, MLX host runtime redirection, or an MLX
test-suite claim.

The complete binary family gate covers all 4,122 discovered current-pinned binary
entries from ``binary.metal``: fifteen 238-entry base shapes and three 184-entry
work-per-thread shapes. The 18 shapes and 11 concrete kernel templates span 24
operators and 25 concrete input/output type pairs across Boolean, signed and
unsigned integer, float16, float32, bfloat16, and complex64 values. Every
selected run emits one selected operator implementation and one kernel, prunes
non-selected operator bodies, and rejects residual template, ``decltype``,
call-operator, or unsupported-placeholder syntax.

Scalar-scalar artifacts materialize ``binary_ss`` and reflect three buffers.
Scalar/vector and vector/vector artifacts, including 2-D and source-default
work-per-thread forms, reflect those buffers plus ``size``. Fixed one-, two-,
and three-dimensional generalized artifacts add exact ``elem_to_loc_1``,
``elem_to_loc_2``, or ``elem_to_loc_3`` index helpers and two stride constants.
The ``gn2`` and ``gn4large`` forms add ``elem_to_loc_2_nd`` plus shape, two
stride, and rank constants. The resulting 4,122 artifacts contain 6,026 exact
materializations and 19,106 reflected resources across exact three-, four-,
five-, and seven-resource ABIs. Explicit and source-default ``N``/``IdxT``
provenance, call-site helper provenance, and a host-owned ``[1, 1, 1]``
workgroup contract are pinned per shape.

Generic repository-scale repairs behind this family map concrete 64-bit vectors
to native ``longN``/``ulongN`` types, rebind dependent free-operator calls only
to already materialized exact helpers, and select non-explicit constructors for
known contextual aggregate conversions while rejecting explicit-only and
ambiguous cases. Bfloat ``min``/``max`` arguments promote to float before typed
result reconstruction. Discarded type-constructor expressions retain argument
evaluation through an unambiguous ``(void)(...)`` form, and scalar Boolean
relational operands receive their C++ integral promotion explicitly. The
focused regressions retain fail-closed behavior outside those proven forms.

Every entry identity, shape, template, operator, exact input/output pair,
semantic family, SHA-256, byte count, materialization contract, and host ABI is
pinned by
[`contracts/binary.metal-roundtrip.json`](contracts/binary.metal-roundtrip.json)
and linked by hash from ``expected-gaps.json``. The prior
[`contracts/binary.scalar-metal-roundtrip.json`](contracts/binary.scalar-metal-roundtrip.json)
remains an exact 238-entry ``ss_`` subset. Required native macOS CI invokes
``xcrun -sdk macosx metal -Werror -c`` across 24 disjoint shards and requires
4,122 non-empty AIR outputs; no source-warning exemption is used. This closes
selected-entry translation, reflection, and native compilation for every
discovered binary instantiation. It is not Metal numerical execution, MLX
host-runtime redirection, or an MLX test-suite claim.

The same selected-entry pipeline translates all 4,122 binary entries to
standalone OpenGL ``main`` artifacts. The schema-v2
[`contracts/binary.opengl-translation.json`](contracts/binary.opengl-translation.json)
contract preserves all 18 shapes, 11 templates, 24 operators, 25 type pairs,
and 6,026 exact materializations while pinning
16,276,504 generated GLSL bytes and 19,106 reflected
resources. Scalar artifacts expose three storage buffers; size-bounded vector
forms add an entry-scoped uniform block. One-dimensional generalized forms add
entry-scoped stride blocks, fixed two- and three-dimensional forms expose stride
storage buffers, and rank-generic forms expose shape and two stride buffers plus
an entry-scoped rank block.

OpenGL cannot implicitly preserve the source's runtime 64-bit buffer indices.
The complete contract therefore records explicit host/runtime bounds of
``[0, 2147483647]`` for ``offset + i``, ``a_idx``, ``b_idx``, ``out_idx``,
``out_idx++``, ``idx.x``, and ``idx.y``. These bounds are declared portability
preconditions, not inferred facts or generated runtime checks; omitting the
relevant proof keeps wide-index translation fail-closed. The generated target
retains exactly the selected operator implementation and one kernel while pruning
all unrelated operator bodies.

Required Linux CI partitions the family into 24 disjoint shards. Every shard
retranslates its exact entries, verifies artifact identity, source/default and
call-site materialization provenance, ``main`` workgroup metadata, and exact
three- through seven-resource host ABI, then compiles with
``glslangValidator --target-env opengl --target-env spirv1.3 -S comp`` and
validates with ``spirv-val --target-env spv1.3``. All 4,122 SPIR-V modules must
be non-empty. This closes discovered binary OpenGL translation, reflection, and
native compiler coverage; it does not claim numerical execution, MLX
host-runtime redirection, or MLX test-suite parity.

The DirectX sibling contract translates all 4,122 discovered entries to
standalone ``CSMain`` artifacts and pins their exact HLSL identities in
[`contracts/binary.directx-translation.json`](contracts/binary.directx-translation.json).
It preserves the same 18 shapes, 11 templates, 24 operators, 25 type pairs,
6,026 materializations, and seven explicit index-range preconditions. DirectX
resource namespaces retain the source buffer coordinates, while nine shapes
that consume ``threads_per_grid`` add ``CrossGLDispatchInfo``. The resulting
21,248 reflected DirectX resources span exact three-, four-, five-, six-, and
eight-resource target interfaces; the rank-generic forms expose shape and
stride buffers, an entry-scoped rank constant block, and generated dispatch
metadata.

Computed-result bfloat ``ArcTan2``, ``LogAddExp``, and ``Power`` paths expand
both operands to float, compute in float, and reconstruct the result in the
exact low-16-bit bfloat register representation with round-to-nearest ties-to-
even. ``Maximum`` and ``Minimum`` expand only for comparison and return the
selected original bfloat payload without requantization. The contract remains
explicit and fail-closed: unrelated bfloat builtins are not admitted, and
storage still requires Shader Model 6.2 native 16-bit types.

Required Windows CI partitions the family into 24 disjoint shards. Every shard
retranslates its exact entries, verifies deterministic identity, source/default
and call-site materialization provenance, ``CSMain`` workgroup metadata, and
exact three- through eight-resource host ABI, then compiles with checksum-pinned
DXC using ``-enable-16bit-types -WX -T cs_6_2 -E CSMain``. All 4,122 DXIL
modules must be non-empty. This closes discovered binary DirectX translation,
reflection, and native compiler coverage; it does not claim numerical
execution, MLX host-runtime redirection, or MLX test-suite parity.

Windows CI compiles the selected HLSL entries with DXC and executes them through
the native loader on Direct3D 12 WARP. Linux CI compiles the GLSL entries with
`glslangValidator` and executes them through the same loader contract on a
surfaceless Mesa OpenGL context. The Square workload maps
`[-3.0, -1.5, 0.0, 2.0, 4.25]` to
`[9.0, 2.25, 0.0, 4.0, 18.0625]`. The ArcCos workload maps
`[-1.0, -0.5, 0.0, 0.5, 1.0]` to the corresponding five float32 arccos values.
Both platforms enforce explicit numerical tolerances. This is numerical
evidence for two selected unary specializations. This numerical evidence
does not cover the other unary operations or dtypes, redirect the MLX host
runtime, or run the MLX test suite.

The checked-in
[`contracts/rms_norm.dispatch.json`](contracts/rms_norm.dispatch.json) fixture
captures 12 distinct dispatch artifacts exercised by the pinned
`python/tests/test_fast.py::test_rms_norm` and `test_rms_norm_grad` workloads.
The forward records cover float32 workgroups of 32, 64, and 128 threads,
float16 and bfloat16 workgroups of 32 threads, and the 1024-thread looped path.
The VJP records cover 32- and 64-thread single-row paths plus the 1024-thread
looped path, each with both concrete values of function constant `20`
(`has_w`). Axis sizes 31, 32, and 33 share the same 32-thread artifact and are
recorded as covered inputs rather than duplicate artifacts.

The repository harness evaluates those finite records against the unchanged
pinned `rms_norm.metal` source and requires 12 deterministic DirectX artifacts.
Every artifact retains its workload inputs, dispatch workgroup count,
host-derived `numthreads` value, concrete function constants, and
`[WaveSize(32)]` enforcement. Windows CI compiles every artifact with official
DXC `cs_6_6`, `-enable-16bit-types`, and warnings as errors. This is complete
translation and compiler coverage for the listed unit-test dispatch variants;
the MLX runtime is not redirected to these artifacts, the kernels are not
executed, and numerical parity is not claimed.

Validate the RMSNorm contract schema, provenance, deterministic identities, and
bounded workload set with:

```bash
.venv/bin/python -m pytest -q -n auto \
  tests/test_mlx_rms_norm_dispatch_contract_fixture.py
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
into a standalone `main` artifact. This existing RMSNorm proof deliberately
does not configure an OpenGL subgroup-width rule, so subgroup provenance and
enforcement metadata remain absent. Linux CI compiles all 24 GLSL artifacts to
OpenGL SPIR-V 1.3 and validates all 24 binaries with `spirv-val`; that result
does not establish the source's 32-lane simdgroup or `simd_sum` semantics. The
bounded LogSumExp proof above exercises the exact-width OpenGL contract without
inflating the RMSNorm claim.

The 24-artifact specialization proof above remains translation and native
compilation evidence only. A separate bounded proof selects the current-corpus
`rmsfloat32` entry at commit
`846d176227a0ac13d2667e58d2bb68b322109ab0` through
[`contracts/rms_norm.native-loader.dispatch.json`](contracts/rms_norm.native-loader.dispatch.json).
That contract fixes a `[32, 1, 1]` workgroup, subgroup width 32, two dispatch
workgroups, and no function-constant value for the forward entry. Entry-scoped
runtime reflection now omits the unreachable VJP-only `has_w` constant while
preserving reachable constants, resolving
[#1795](https://github.com/CrossGL/crosstl/issues/1795). The selected template
materializes one `rms_single_row<float, RMS_N_READS>` specialization from four
reachable specializations while pruning 168 unrelated candidates.

The generated HLSL artifact is 3,486 bytes with SHA-256
`f03d8c3c1df2256e5c867bfd235e57b66d68a1c6e3c3c04701a581d8ef7b3e67`;
it retains `[WaveSize(32)]` and compiles as `cs_6_6` with
`-enable-16bit-types`. The generated GLSL
artifact is 4,393 bytes with SHA-256
`3180aba83b64add0ae3c2d471b9297eb5bada4c4ff2bd5c91a3db3698cf0df78`.
Its explicit target-scoped 32-lane software subgroup lowers scalar
`WaveActiveSum`, emits six `OpControlBarrier` instructions and no
`OpGroupNonUniform` instruction, and passes `glslangValidator` plus
`spirv-val`. Default OpenGL generation remains on the hardware-subgroup path.

The six-buffer native-loader ABI packages float32 input, weights, and output,
plus 16-byte scalar blocks for epsilon, axis size, and weight stride. The
numerical workload runs two deterministic float32 rows of axis size 32 with 32
weights and compares 64 outputs against
`x * w * rsqrt(mean(x * x) + epsilon)` at `3e-5` absolute and relative
tolerance. CI requires the same package and readback on Direct3D 12 WARP and
Mesa software OpenGL. This is bounded execution evidence for one forward
float32 workload; it does not redirect the MLX host runtime, run the MLX test
suite, or cover complete RMSNorm runtime parity. In particular, that forward
proof does not cover VJP, looped, float16, or bfloat16 entries, other axis
sizes, or the remaining host/device dispatch space.

A sibling current-corpus VJP proof selects `vjp_rmsfloat32` through
[`contracts/rms_norm_vjp.native-loader.dispatch.json`](contracts/rms_norm_vjp.native-loader.dispatch.json).
It fixes one float32 row of axis size 32, `has_w=true` through function constant
ID `20`, a `[32, 1, 1]` workgroup and 32-lane subgroup, one dispatch workgroup,
and the two explicit 0-through-31 index-range assertions needed by the bounded
input and cotangent views. Materialization selects
`vjp_rms_single_row<float, RMS_N_READS>` from four reachable specializations
while pruning 168 unrelated candidates.

The generated HLSL is 6,795 bytes with SHA-256
`7c1fe2a3c5f6d883b11b3fb17511663ebb3ead2a0931611229930c3f07035c9f`.
It concretizes `has_w=true`, retains `[WaveSize(32)]` and four
`WaveActiveSum` calls, and compiles as `cs_6_6` with
`-enable-16bit-types`. The generated software-subgroup GLSL is 7,771 bytes
with SHA-256
`2112adeb6c1693fa42c48fe3013cd57637f34a9393c0d468b547ed06ab42cf73`.
It retains deferred OpenGL specialization constant `20`, emits six control
barriers with no hardware-subgroup extension or SPIR-V group-nonuniform
instruction, and passes `glslangValidator` plus `spirv-val`.

RMSNorm VJP exercises a canonical runtime row loop. The explicit software
subgroup path accepts that loop only after proving its initializer and bound
workgroup-uniform from `gl_WorkGroupID` and read-only scalar blocks; lane-varying
inputs, bound mutation, unresolved calls, and `break`, `continue`, or `return`
remain fail-closed. Default OpenGL generation remains on the hardware-subgroup
path.

The ten-resource native-loader ABI contains float32 `x`, `w`, `g`, `gx`, and
one-group `gw` buffers plus float32 epsilon and uint32 axis-size, weight-stride,
row-count, and rows-per-group scalar blocks. The deterministic request compares
all 32 `gx` and 32 `gw` values at `7e-5` absolute and relative tolerance. A
local Linux arm64 llvmpipe EGL run passed deferred GLSL-to-SPIR-V execution with
maximum absolute errors below `3.54e-8` for `gx` and `3.40e-8` for `gw`; CI
requires the same package and numerical readback through Direct3D 12 WARP and
Mesa software OpenGL.

This proof is intentionally one row and one group, so the kernel's group-local
`gw` result is also the final host-reduced weight gradient. It does not cover
multi-row weight reduction, `has_w=false`, float16 or bfloat16, looped entries,
other axis sizes, MLX host redirection, or the full MLX test suite.

`fence.metal` emits no DirectX, OpenGL, or Vulkan target artifact. The harness
requires the target-specific structured diagnostics and the exact requested
atomic-fence operands under #1537 instead of accepting generated barrier text as
semantic evidence.
Future scouts should add issue-backed blockers only when there are
concrete repros. Host runtime integration gaps should be handled in repository
integration code or downstream runtime adapters, not hidden as shader
translation successes.
The full GEMV Vulkan gate materializes all 226 source specializations and emits
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
resolves CrossGL/crosstl#1814 and removes
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
resolves CrossGL/crosstl#1610 for this contract without claiming that scalar
element expressions require matrix result types.

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

Exact high-budget `fp_quantized.metal` replays enable lane-local
cooperative-matrix lowering explicitly through both code-generation factory
paths, `crosstl.project.pipeline.get_codegen` and
`crosstl._crosstl.get_codegen`; this option is not yet available through project
configuration.

Three measured intermediate replays document translation progression and are
not current-boundary claims. In the first replay, DirectX ran for 292.953
seconds and OpenGL for 286.175 seconds before both reported
`project.translate.metal-local-type-unresolved` for local type `vec_w`, whose
extent remained `tn * bytes_per_pack`. In the second replay, DirectX ran for
286.540 seconds and OpenGL for 283.718 seconds. Both reached extent
`(2) * bytes_per_pack`, proving that `tn` had resolved to `2`. In the third
replay, DirectX ran for 418.540 seconds and OpenGL for 364.117 seconds; each
materialized 606 function specializations before branch-insensitive
private-pointer analysis reported a false `view-out-of-bounds` result for
`qouter_float_2_8_4.w`.

The implementation sequence establishes reusable contracts for function-local
struct hoisting, concrete `constexpr` local extents, and defaulted zero-argument
helper materialization. These contracts are backend-independent materialization
behavior rather than MLX-specific rewrites. The completed replay demonstrates
progression through all three contracts for this source specialization.
CrossGL/crosstl#1567 remains open globally because this source-specific evidence
does not establish its complete function-local typedef scope.

DirectX and OpenGL branch pruning, together with the project translation
regression, resolve CrossGL/crosstl#1829. The completed exact replay proves
progression past the earlier false `qouter_float_2_8_4.w` range result.

In the completed replay, DirectX ran for 429.507 seconds, materialized 606
function specializations with no unsupported specializations, and reported
`project.translate.directx-workgroup-pointer-unsupported`. The missing
capability is `directx.workgroup-pointer-lowering`; function
`BlockMMA_float_float_16_32_32_1_2_false_true_36_36__mma`, parameter `As`, stops
with reason `dynamic-control-flow-reassignment` and message `DirectX cannot
preserve workgroup pointer reassignment for 'As' across nested control flow`.
CrossGL/crosstl#1518 covers the required HLSL resource and `groupshared` alias
representation, reassignment and nested-alias semantics, and structured
rejection.

OpenGL ran for 417.522 seconds, materialized the same 606 function
specializations with no unsupported specializations, and reported
`project.translate.opengl-workgroup-pointer-unsupported`. The missing capability
is `opengl.workgroup-pointer-lowering`; parameter `dst_` stops with reason
`bare-pointer-expression` and message `OpenGL cannot emit a workgroup pointer as
a first-class value: dst_`. This target boundary spans two existing contracts:
CrossGL/crosstl#1544 covers pointer-bearing aggregate members and constructors,
including `QuantizedBlockLoader`, while CrossGL/crosstl#1671 covers concrete
workgroup backing provenance through helper parameters. It is not classified as
a shared DirectX/OpenGL boundary.

Each target report contains one failed artifact/provenance record, zero
translated artifacts, and one error; no target artifact was emitted. Native
validation was not attempted because there is no artifact. MLX host runtime
integration and execution were not attempted, and numerical parity was not
evaluated. CrossGL/crosstl#1546 remains open for the broader byte-address
provenance contract across pointer reinterpretation, but it is not the current
exact boundary. Earlier replays also established progression past
`qdot_float_16_4.x_thread` and its `unprovable-view-offset` boundary under
CrossGL/crosstl#1826, as well as the transitive local `constexpr`
materialization contract tracked by CrossGL/crosstl#1824.

A shared reduced CrossGL fixture exercises the resolved partition contract with
the required readback `[100, 101, 102, 103, 200, 201, 202, 203]`.
[GitHub Actions run 29641172600](https://github.com/CrossGL/crosstl/actions/runs/29641172600)
produced this exact readback on `windows-latest` through Direct3D and on
`ubuntu-latest` through OpenGL. The passing steps were
`Prove Direct3D private-pointer partition writeback` and
`Prove OpenGL private-pointer partition writeback`, respectively.

An independent Metal fixture, `private_pointer_word_view.metal`, initializes a
local struct with two 32-bit words, reads its eight bytes through a const
thread-local byte view, and computes the order-sensitive checksum
`sum(byte[index] * (index + 1))`. Its required readback is `[204]`. The Metal
source compiles locally as Metal 3.2 with Apple metal `32023.918`.
[GitHub Actions run 29649620337](https://github.com/CrossGL/crosstl/actions/runs/29649620337)
produced the exact `[204]` readback on `windows-latest` through Direct3D and on
`ubuntu-latest` through OpenGL. The passing steps were
`Prove Direct3D local-struct byte-view native readback` and
`Prove OpenGL local-struct byte-view native readback`, respectively. Both tests
required their native runtime and reported zero mismatches with zero absolute
and relative tolerance.

These reduced fixture-level proofs do not establish complete MLX artifact
translation, full MLX host runtime integration, full MLX test-suite execution,
or numerical parity for MLX workloads.

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
