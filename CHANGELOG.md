# Changelog

All notable changes to CrossTL are documented in this file.

## [3.2.0] - 2026-08-14

### Added

- Explicit target-scoped DirectX 32-lane software subgroup lowering for supported scalar shuffle-down operations, using flattened `SV_GroupIndex` identities, bounded groupshared scratch storage, synchronized reads, and exact calling-invocation fallback without hardware wave-lane reads.
- Native Direct3D and OpenGL loader target adapters with a stable generation registry, structured diagnostics, artifact verification, SPIR-V specialization, and translated-device proofs.
- Exact native runtime variant dispatch packages that bind ready variants to copied target artifacts, verified loader descriptors, and deterministic C++17 execution wrappers, with Python and C++ lookup interfaces.
- Bounded deferred native compilation: a closed versioned request contract, verified input packaging with deterministic provenance, a success-only output cache keyed by request and toolchain identity, and execution through native runtimes (HLSL to DXIL, GLSL to OpenGL SPIR-V).
- Runtime variant binding to deferred native compilation, deriving bounded DirectX and OpenGL compilation requests only after exact registry, package, source, descriptor, target, execution, and specialization identity checks.
- Execution-boundary verification of native artifact identity before dispatch.
- Resumable project translation checkpoints with a durable contract, interruption recording, and checkpoint identity that covers configured limits.
- Bounded, process-isolated project translation workers with deterministic report and checkpoint coordination, lazily streamed plans, transactional artifact pair publication, coordinated concurrent publication, typed worker failure coordinates, interrupted-worker termination, and per-artifact wall-clock timeouts, exposed through both the API and CLI.
- Multiple project entry artifacts per source, a backend-neutral project source entry-point discovery contract with Metal enumeration, and opt-in discovered-entry expansion for selected sources recorded in reports, artifact matrices, and checkpoints.
- Pinned MLX native execution evidence: a binary kernel executed on DirectX and OpenGL with exact device readback, quantized OpenGL lowering and quantized gather provenance proofs, and bounded LayerNorm host dispatch integrated into the MLX porting frontier.
- Current-pinned MLX LayerNorm axis-32 native-loader packages with exact eight-resource ABIs, Direct3D 12 WARP and Mesa EGL numerical CI, deterministic HLSL/GLSL identities, and an upstream-derived dispatch contract.
- Current-pinned MLX LayerNorm VJP axis-32, one-row, `has_w=true` native-loader execution with exact eight-resource ABI, concrete DirectX and deferred-specialized OpenGL paths, and numerical `gx`/`gw` readback on WARP and Mesa.
- Current-pinned MLX RMSNorm VJP axis-32, one-row, `has_w=true` native-loader execution with an exact ten-resource ABI, concrete DirectX and deferred-specialized OpenGL paths, and numerical `gx`/`gw` readback on WARP and Mesa.
- Current-pinned MLX 512-thread float32 dot-product execution through Direct3D 12 WARP and a distinct sixteen-subgroup software OpenGL artifact on Mesa, with exact four-resource ABI, SPIR-V structure, artifact identity, and numerical readback evidence.
- Current-pinned MLX block Softmax execution for two axis-32 rows and one axis-2049 row through Direct3D 12 WARP and explicit software-subgroup OpenGL on Mesa, with exact three-resource ABIs, deterministic guarded and software artifact identities, SPIR-V structure, and stable float32 numerical references.
- Current-pinned MLX `argmin_float32` and `argmax_float32` execution for two axis-32 rows through Direct3D 12 WARP and explicit software-subgroup OpenGL on Mesa, with exact signed/unsigned 64-bit resource ABIs, deterministic tie-breaking readback, and separate aggregate and Metal limitations.
- Current-pinned MLX one-pass `sdpa_vector_float_64_64` execution through Direct3D 12 WARP and deferred-specialized software-subgroup OpenGL on Mesa, with a host-derived 1,024-thread dispatch, exact 19/18-resource ABIs, six false function constants, deterministic artifact identities, and 64-value numerical readback evidence.
- Current-pinned MLX `fft_mem_256_float2_float2` OpenGL execution through deferred SPIR-V on Mesa, with deterministic GLSL identity, 21 specialization constants, exact four-resource vector ABI, validated barrier structure, and 256-complex-value impulse readback evidence.
- Current-pinned MLX `gemv_t_float32_bm1_bn2_sm8_sn4_tm4_tn4_nc0_axpby0` execution for a 32x32 float32 vector-matrix workload through Direct3D 12 WARP and two-subgroup software OpenGL on Mesa, with exact 15-resource ABIs, deterministic HLSL/GLSL identities, validated SPIR-V structure, and 32-value numerical readback evidence.
- Current-pinned MLX `mxfp4_quantize_dequantize_float_gs_32_b_4_hgs_false` execution for 32 exact FP4 E2M1 values through Direct3D 12 WARP and one-subgroup software OpenGL on Mesa, with a host-derived dispatch contract, exact reflected resources, deterministic HLSL/GLSL identities, validated SPIR-V structure, and zero-tolerance numerical readback.
- Deferred-specialization ABI packages retain a ready, actionable JSON runtime variant registry while marking the unsupported generated C++ registry header unavailable with an explicit reason.
- Deferred Native Compilation CI workflow covering a three-platform contract matrix plus native Direct3D 12 and software OpenGL compile-and-dispatch jobs with pinned DXC verification and required device readback.

### Improved

- DirectX software subgroup admission fails closed unless one bounded compute entry has concrete width-compatible dimensions, a supported payload and operation, unambiguous helper identity, logical invocation identity, explicit out-of-range policy, and statically uniform barrier control flow; relative source indexing is guarded before unsigned addition.
- HLSL host-reflection collision checks now distinguish CBV, SRV, UAV, and sampler register namespaces, allowing legal coordinates such as `b0` and `t0` while continuing to reject duplicate bindings within one namespace.
- Explicit 32-lane OpenGL software subgroups now partition bounded workgroups of up to 1,024 invocations into independent logical subgroups and admit narrow typed-identity-masked divergent `WaveActiveSum`, `WaveActiveMin`, and `WaveActiveMax` assignments while all generated barriers remain workgroup-uniform; conditional shuffle and ambiguous control flow still fail closed.
- Explicit OpenGL software subgroups synchronize canonical subgroup-ID-strided runtime loops across complete workgroup rounds when the bound is workgroup-uniform, the stride equals the logical subgroup count, and exactly one supported masked reduction is present; unsafe declarations, mutation, escaping control flow, and multiple collectives remain fail-closed.
- HLSL and GLSL host reflection represents stored Boolean buffers with their portable physical uint32 ABI while preserving Boolean shader-level and specialization-constant types.
- GLSL host reflection records exact standard scalar/vector physical widths and `std140`/`std430` alignment, including tight `vec2`/`vec4` storage and padded `vec3` array stride; native dispatch accepts supported tight vectors and rejects padded uploads rather than guessing.
- Metal analysis performance: lexical scope lookups are indexed and repeated whole-tree traversals eliminated.
- OpenGL resource specialization includes the storage pointer view layout in specialization identity and generated helper names, so direct and reinterpreted calls against the same root receive distinct deterministic helpers regardless of call order.
- Explicit OpenGL software subgroups accept uniquely identified helpers only when calls are direct, unconditional, top-level, and entry-owned; resource-specialized calls may reuse only identical-base workgroup proofs whose validated intervals cover the narrower request.
- Explicit OpenGL software subgroups admit canonical runtime loops only when integer initializers and bounds are proven workgroup-uniform through constants, scalar blocks, workgroup builtins, and conservative local dataflow; lane-varying state, mutation, escaping control flow, and unresolved calls remain fail-closed.
- Explicit OpenGL software subgroups admit direct top-level helper calls inside proven canonical workgroup-uniform positive-to-zero halving loops using the integral-equivalent `value > 0` or `value >= 1` bound with `/= 2` or `>>= 1`; wider bounds, nested calls, ambiguous identities, unsafe updates, mutation, and escaping flow remain fail-closed.
- Native reflection and runtime requests support exact scalar `int64_t` and `uint64_t` structured/storage buffers and single-member constant/uniform blocks, including range-checked little-endian packing, readback, stride inference, and 16-byte scalar-block alignment.
- HLSL compute stages are inferred from `numthreads` attributes, keeping deferred interface reflection aligned with DXC entry-point selection.
- Project artifact report identities stay repository-relative on Windows by preferring lexical project-root containment.
- Per-artifact time limits are applied to MLX full-corpus scouting so stalled units produce structured diagnostics and durable checkpoint records.
- Support matrix, generated roadmaps, and project-porting documentation regenerated for the new DirectX and OpenGL coverage.
- Test suite grown to over 17,500 collected tests (from 13,485 at v3.0.0).
- pre-commit tooling updated (ruff 0.15.22 to 0.16.2, isort 9.0.0b1 to 9.0.0b2).

### Fixed

- Current-pinned multidimensional MLX GEMV no longer relies on WARP physical-wave lane topology: its DirectX reduction now uses deterministic logical 32-lane software subgroups, fixing the observed replacement of logical lanes 5–8 by physical lanes 21–24.
- DirectX native 16-bit Metal `as_type` lowering uses exact `asfloat16`/`asint16`/`asuint16` reinterpretation by default; an explicit source-scoped widening mode reconstructs binary16 payloads with integer IEEE-754 masks and keeps logical `float16_t` arithmetic, sign application, returns, and consumers in float32. Scalar native `int16_t`/`uint16_t` arithmetic now also follows Metal/C++ integer promotion before shifts, preventing FP8 scale-bit construction from truncating `<< 23` to a 16-bit shift and eliminating current-pinned MXFP4’s signed-zero collapse on WARP.
- DirectX compute `gl_SubgroupID` lowering assigns a synchronized, wave-uniform physical ID in multi-wave workgroups instead of dividing each lane's flattened `SV_GroupIndex`; proven one-wave workgroups retain the valid quotient fast path.
- OpenGL bitcasts involving widened binary16 values preserve scalar and packed-half payloads through `packHalf2x16`/`unpackHalf2x16` instead of incorrectly reinterpreting them as float32 payloads.
- Metal conversion analysis treats parenthesized returns as control syntax, resolves proven local conditional aliases to concrete constructor factories without guessing unresolved branches, and recognizes qualified `round`; OpenGL emits exact single-evaluation float32 `isfinite` and `signbit` bit tests while preserving user overloads and structured fail-closed diagnostics.
- OpenGL resource specialization propagates provably dead default-null storage pointers through direct forwarding chains, decodes exact materialized Metal `vec<T,N>` constructor names, and emits every collision-safe overloaded specialization body; unresolved or observable pointer uses remain fail-closed.
- DirectX relative wave-shuffle lowering now offers an explicit target-scoped `self` policy that selects only valid source lanes and returns the calling lane for out-of-range down/up/xor reads; the default undefined policy and unrelated artifact identities remain unchanged.
- OpenGL bounded index narrowing now tracks scalar and vector-component ranges across declarations, branches, loops, casts, and mutation boundaries, normalizes supported 64-bit indices once, and fails closed when no narrowing proof exists.
- OpenGL scalar conversion contexts, explicit source truthiness lowering, narrow integer initializer contracts, and `isnan`/`isinf` result-type inference.
- OpenGL boolean arithmetic compound assignments and trailing-zero builtin lowering, with structured diagnostics for unsupported operand contracts and target profiles.
- OpenGL private arrays are preserved in resource-specialized callers, redundant same-view storage pointer casts are treated as idempotent, and reinterpreted storage byte bases are forwarded explicitly so nested helpers keep pointer arithmetic without capturing caller-local expressions or double-scaling offsets.
- DirectX aggregate initialization: destination type information is preserved while probing typed-buffer atomic expressions, omitted structure and fixed-array fields are recursively value-initialized, and untyped array literals are rejected with a structured diagnostic instead of ambiguous HLSL brace expressions.
- Metal helper calls preserve mutable resource pointer offsets in both DirectX and OpenGL output.
- Source-instantiated Metal template helpers with overloaded names defer to signature-aware materialization, preventing scalar call sites from selecting vector overloads.
- Direct3D shutdown with live pipelines and unverified HLSL includes now fail closed, and OpenGL adapter object ownership is hardened with caller-controlled context lifetime for injected contexts.
- Worker timeout exceptions are distinguished from job deadlines so a timeout raised inside translation code remains a typed worker failure.
- Python 3.8 and 3.9 compatibility preserved across entry discovery and deferred compilation tests.
- Deferred native-loader proof annotations remain importable on Python 3.8 and 3.9.

---

## [3.1.0] - 2026-07-19

### Added

- Portable runtime execution graphs, deterministic runtime variant registries, and request generation from native loader descriptors.
- Generated native loader execution ABI packages with deterministic C11 and C++17 host integration artifacts.
- Native compute runtime drivers and project-runner adapters for Direct3D, headless OpenGL, and Vulkan.
- Initialized read-write resources, shared native allocation views, aggregate byte views, and private-array partition views for native execution.
- Cooperative-matrix types and fragment contracts in the shared IR, including Vulkan KHR cooperative-matrix SPIR-V lowering foundations.
- Pinned MLX project translation and validation across DirectX, OpenGL, and Vulkan, including reduced-frontier and full-corpus workflows.

### Improved

- Metal project materialization for templates, overloads, references, pointer inference, constexpr propagation, wide vectors, and aggregate return types.
- DirectX lowering for fixed arrays, 16-bit arithmetic, overload identity, resource relocation, specialization constants, and runtime dispatch.
- OpenGL lowering for storage aliases, interface-backed structs, specialization constants, scalar conversions, reserved identifiers, and runtime-derived globals.
- Runtime manifests and host handoff metadata now track selected entry points, device readiness, execution resources, and native loader package identities.
- MLX compiler-frontier accounting, native validation, and runtime proofs now cover more kernels and fail closed on unsupported contracts.

### Fixed

- Source-language arithmetic conversion order, explicit constructor handling, boolean compound assignments, conditional expressions, and file-scope constant preservation across target backends.
- Metal alias-template, receiver, reference-accessor, addressed-pointer, and overload-resolution edge cases encountered in project-scale translation.
- DirectX entry-point validation, fixed-array initialization, aggregate conditionals, minimum-precision division and remainder, and unrepresentable texture-offset diagnostics.
- OpenGL block-name collisions, partial vector initialization, fixed-array loops, static-member layout leakage, and correlated private pointer views.
- SPIR-V helper return lowering, empty aggregate typing, unresolved generic storage analysis, and cooperative-matrix memory operations.

---

## [3.0.0] - 2026-06-23

### Added

- Target-only WebGL/WebGL2 GLSL ES backend with vertex and fragment stage output.
- Target-only WebGPU/WGSL backend with vertex, fragment, and compute output.
- DirectX target profile aliases (`dx11`, `dx12`, `d3d11`, `d3d12`) all resolving to HLSL source output.
- MLX project porting integration with demos, CI workflow, and full-corpus validation.
- Runtime host integration pipeline: loader scaffolds, adapter descriptors, manifest metadata, and execution adapters.
- Project-scale shader porting pipeline with runtime readiness checks and blocked-unit tracking.
- Metal struct-template materialization engine with instantiation, specialization, SFINAE-overloaded member lowering, and budget scaling.
- Metal subgroup/SIMD intrinsic lowering to CrossGL Wave ops.
- Metal atomic device lowering to CrossGL atomic intrinsics.
- Metal threadgroup-local array hoisting to GLSL shared globals.
- Runtime parity executors and verification harness for translated GPU artifacts.
- WGSL resource binding, storage buffer, and texture sampler object model support.
- Naga-based WGSL validation for generated project artifacts.
- SPIR-V artifact assembly and validation pipeline.
- Host runtime detection for WebGPU, OpenCL, D3D11, Rust wgpu, and game engine runtimes.
- Open-source project porting demos with third-party notice coverage and failure summary reporting.
- Runtime loader manifest generation and package inspection commands.
- Host integration handoff bundles with blocked-step reporting.
- Integer sampled texture lowering across backends.
- GLSL specialization constant lowering for target backends.
- Metal function constant lowering for SPIR-V output.
- Backend-agnostic runtime manifest metadata for cross-target artifact planning.

### Improved

- Metal preprocessor: full-source comment masking, functor lowering, template budget scaling, deterministic member lowering, and constexpr array extent recognition.
- HLSL compute codegen: value-builtin lowering, subgroup-to-wave translation, semantic validation, and program-scope constant output.
- GLSL codegen: subgroup builtins, specialization constants, vertex layout location mapping, storage image lowering, and fragment invocation density modeling.
- SPIR-V codegen: storage buffer overload resolution, 64-bit layout validation, bool compute interfaces, access-chain typing, and gather offset vector operands.
- DirectX codegen: groupshared hoisting, resource binding allocation, fragment varying semantics, typed buffer indexed writes, and loader profile metadata.
- OpenGL codegen: Metal template specialization, buffer parameter lowering, stage input struct mapping, uniform block member disambiguation, and GLSL 330 resource bindings.
- OpenCL reduce target artifact generation, diagnostics, and helper contract reporting.
- Rust-GPU: vertex input lowering, compute construct lowering, storage buffer declarations, and Option value pre-lowering.
- Mojo: GPU kernel extraction, vector overload resolution, compute builtin parameterization, and sampled texture binding lowering.
- Slang: resource name collision avoidance, default parameter preservation, texture size query casting, and compute entry lowering.
- WGSL: structured storage buffers, constant buffer block lowering, derivative intrinsic mapping, image2D storage textures, and reserved identifier escaping.
- Project translation: unresolved-construct guardrails, source provenance tracking, macro variant classification, include resolution, and feature root scanning.
- Support matrix tooling: target alias metadata, target-only source rejection documentation, and backend conformance edge coverage.
- Split target backend registration from native source frontend availability in the backend registry and support matrix.
- Test suite grown to 13,485 collected tests (from ~8,000 at v2.0.0).

### Fixed

- Over 100 targeted backend bug fixes across CUDA, HIP, Metal, GLSL, HLSL, SPIR-V, WGSL, OpenCL, Rust-GPU, Mojo, and Slang.
- Metal residual-template false positives from struct member templates.
- SPIR-V storage buffer pointer overload inference and access-chain index typing.
- WGSL vector narrowing, entry struct varying locations, reserved identifier escaping, and uniform scalar array layout.
- DirectX vertex input semantics, scalar fragment targets, binding allocation, and groupshared hoist aliases.
- HLSL compute value-builtin leakage, scalar splat swizzle lowering, and array semantic allocation.
- OpenGL bfloat16 asuint lowering, GLSL 110 interface qualifiers, and gl_FragColor reserved local conflicts.
- Vulkan matmul builtin vector stores, complex helper call argument types, and private scalar initializer types.
- HIP scalar kernel params, bit-extract kernel loop preservation, and host main artifact generation.
- CUDA vector-add DirectX lowering and OpenGL uniform name stabilization.
- Metal-to-DirectX compute resource output and constant buffer member SPIR-V access.
- Mojo GPU builtin lowering contract and vector-add Metal/SPIR-V artifacts.
- Slang compute entry lowering to OpenGL and struct resource name collisions.
- WebGL dynamic sampler array helper calls and unsupported sampler resource rejection.
- GLSL usampler texelFetch SPIR-V lowering and fragment output name handling.
- Python 3.8 collection type compatibility in the project pipeline.

---

## [2.0.0] - 2026-06-01

### Breaking Changes

- **CrossGL language expanded**: New constructs (generics, traits, algebraic enums, pattern matching, do-while, double-precision types) mean `.cgl` files written for 2.0 may not parse under 1.x.
- **AST restructured**: `ASTNode` gained `child_nodes()`, `walk()`, and `bind_parent_links()` methods. `ShaderNode.stages` is now a `StageMap` wrapper instead of a raw `dict`.
- **Internal imports changed**: Direct imports like `from crosstl.translator.codegen import GLSL_codegen` are no longer supported. Use the registry (`crosstl.translator.codegen.registry`) or the public `crosstl.translate()` API.
- **Codegen `generate()` contract changed**: Backend code generators now expect the expanded AST structure with new node types.

### Added

- **Validation layer** (`crosstl/translator/validation.py`): 2700+ lines of texture intrinsic, stage, and resource validation.
- **Language specification extractor** (`crosstl/translator/language_spec.py`): Machine-readable snapshot of the CrossGL language surface.
- **Plugin/registry system**: Backends self-register via `BackendSpec`/`SourceSpec`, enabling third-party extensions.
- **Shared codegen utilities**: `enum_utils`, `match_utils`, `generic_function_utils`, `generic_struct_utils`, `vector_arithmetic`, `image_access_contracts`, `resource_arrays`, `resource_query`, `stage_utils`.
- **New language features**: Generics (`generic<T>`), traits, algebraic enums with data, pattern matching (`match`/`case`), guard clauses, do-while loops, double-precision types (`dvec2-4`, `dmat2-4`).
- **Stage coverage**: Full support for compute, geometry, tessellation, mesh/task, and ray-tracing stages across applicable backends.
- **Support matrix tooling**: Automated feature tracking (`support/features.json`), CI-driven matrix validation, and issue synchronization.
- **Documentation**: Sphinx-based docs with per-backend API reference, architecture guide, and support matrix.

### Improved

- All 9 backends (GLSL, HLSL, Metal, SPIR-V, CUDA, HIP, Mojo, Rust, Slang) massively expanded — 10-20x more generated code surface handling new language features.
- CI expanded to cross-platform (Linux/Windows/macOS), multi-Python (3.8-3.13), per-backend test matrices with JUnit XML output.
- Test suite grew to 7,983 passing tests (from ~200 at 1.0.0).

### Fixed

- Fragment output name handling in GLSL codegen.
- Vector constructor declarations in CUDA/HIP parsers.
- Switch-default ordering preservation.
- Texture comparison function translation across backends.

### Removed

- `build/` directory (stale artifact).
- `.agent-coordination/` directory (obsolete automation scaffolding).
- `experimental/` backend directory.
- `pytest` removed from runtime `install_requires` (it was incorrectly listed as a runtime dependency).

---

## [1.1.0] - 2025-12-15

### Added

- Backend registry and source registry for plugin-style backend loading.
- Slang backend (lexer, parser, codegen, preprocessor).
- Mojo and Rust backends.

### Improved

- Refactored shader translation pipeline.
- Backend integration consolidated.

---

## [1.0.0] - 2025-10-01

Initial release with:

- CrossGL intermediate representation language.
- Translation targets: Metal, DirectX (HLSL), OpenGL (GLSL), Vulkan (SPIR-V), CUDA, HIP.
- Bidirectional translation (native → CrossGL and CrossGL → native).
- Basic vertex, fragment, and compute shader stage support.
