# Changelog

All notable changes to CrossTL are documented in this file.

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
