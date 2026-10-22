# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them.

## [Unreleased]

### Breaking Changes

- CrossTL requires Python 3.11 or later. Python 3.8, 3.9 and 3.10 have reached end of life and are no longer supported.
- CrossGL syntax can change between releases until CrossGL expresses everything its targets can. In this release, debug prints are `printf(format, values...)` instead of `debugPrintfEXT`, comparison samplers have their own `comparison_sampler` type, and HLSL and Metal textures import as separate textures and samplers.
- A Metal, CUDA, HIP or OpenCL kernel whose results depend on a threadgroup size that the host picks at dispatch is refused for HLSL, GLSL, SPIR-V and WGSL output, which used to fix the size at 1, 1, 1. Pass the dispatch size as `source_options={"threadgroup_size": (x, y, z)}` or in a project workgroup-size rule.
- A CrossGL function body that does not parse is refused; it used to become an empty block. An untyped `var` whose type cannot be inferred from its initializer, or that has no initializer, is refused instead of being declared `float`.

### Added

- A coverage report. CI translates a probe for every feature in each language's pinned upstream feature list, checks the result with that language's reference compiler or validator, and publishes the scores and every gap to [COVERAGE.md](https://github.com/CrossGL/crosstl/blob/coverage/COVERAGE.md).
- Texel buffers (`Buffer<T>`, `RWBuffer<T>`, `samplerBuffer`, `imageBuffer`) translate between HLSL, GLSL, Metal and SPIR-V.
- DXR ray-tracing stages lower to Vulkan SPIR-V and GLSL, with hit attributes, ray payloads and callable data. HLSL output marks ray payloads with `[raypayload]` and their access qualifiers.
- Shader debug prints translate to SPIR-V and Metal.
- GLSL geometry stages translate to HLSL's stream form.
- Non-uniform resource indices translate to GLSL, Metal and GLSL ES.
- HLSL structured-buffer counters (`IncrementCounter` and `DecrementCounter`) work on Metal, GLSL, CUDA and HIP through an explicit counter buffer.
- Built-in values can be written in their GLSL, HLSL or WGSL spelling in CrossGL and translate to every target.
- Untyped `var name = expr;` declarations, including in `for` loop initializers and at shader scope, take their type from the initializer.

### Improved

- Every supported Python version, 3.11 through 3.15, is tested.
- Unsized HLSL resource arrays get their size on Metal and OpenGL from bounded dynamic indexes as well as constant ones, so translated shaders no longer read past the array.
- HLSL texture samples return the texture's element type, and storage-image loads and stores use four-component texels as GLSL defines them.
- CUDA and HIP buffer elements such as `float3` and structs with vector members keep their host layout on every target.
- SPIR-V modules for mesh, task and ray-tracing stages, which need SPIR-V 1.4, declare storage buffers in the `StorageBuffer` storage class and pass `spirv-val`.
- GLSL stage inputs and outputs are named so the stages link, and never redeclare a global.
- Metal import and graphics code generation are faster on large kernels.

### Fixed

- The parser no longer hangs on an unterminated shader declaration, and parses `if`, `else` and loop bodies without braces.
- HLSL output renames GLSL and Metal identifiers that HLSL reads as keywords, such as `line`, `point`, `vector` and `Buffer`.
- Metal SIMD-group and quad-group functions declared behind `METAL_MAYBE_UNDEF` are recognized.

---

## [3.4.0] - 2026-10-07

### Added

- Source arithmetic profiles for binary32 addition, subtraction, multiplication, division, remainder, precise arctangent and logarithms. DirectX, OpenGL and Metal output get native controls for rounding, subnormals, signed zeros and non-finite values.
- Thread-grid dispatch regions, fixed partial workgroups, Boolean subgroup votes and software subgroup products. Unsupported collective control flow is reported as an error.
- Native runtime support for Boolean, binary16, bfloat16, byte and 64-bit integer storage, shared allocation views and immutable DirectX constants.
- Checked integer metadata preconditions carried through project artifacts, reflection and runtime packages.
- MLX integration demos for array creation, copies, casts, reductions, indexing, scatter and random generation through translated runtime packages.
- Package metadata lives in `pyproject.toml` with a pinned Flit build backend. Releases install the wheel and the source distribution in isolation and verify the published PyPI archives before the GitHub release is created.

### Improved

- Precise Metal square root, logarithm, power and `atan2` keep signed zeros, special values and underflow behaviour under explicit operand profiles.
- HLSL, GLSL and Metal output keep integer promotions, vector initialization, byte narrowing, binary16 selection, bfloat rounding, atomic destinations and argument evaluation order.
- Source maps no longer guess line locations. Only artifacts that preserve lines carry line mappings.
- MLX project files moved to `demos/integrations/mlx`.

### Fixed

- Project metadata requirements apply only to the source entry that declares them.
- Unreferenced out-of-line Metal call operators no longer block kernel translation.
- DirectX compilation resolves relative includes through long Windows paths.
- Software subgroup lane reads, shuffle-and-fill and guarded inputs keep their semantics in HLSL and GLSL output. Metal output uses shuffles for dynamic lane indices.
- HLSL scalar-to-vector construction evaluates increment and decrement operands once.

---

## [3.3.0] - 2026-10-01

### Added

- Binary32 fused multiply-add with explicit rounding and subnormal profiles.
- Native Metal execution through runtime packages and loader descriptors, and dispatch-limit checks in the DirectX, OpenGL and Vulkan drivers.
- Translation of every kernel in the pinned MLX unary, copy, binary and reduce families to DirectX and OpenGL, and Metal round trips of the copy, binary, reduce and quantized families. Selected kernels, among them LayerNorm, RMSNorm, Softmax, SDPA, FFT, GEMV, argmin, argmax and MXFP4, are checked numerically on Direct3D 12 WARP and Mesa.
- Explicit 32-lane software subgroups for DirectX and OpenGL compute. Generated GLSL states the subgroup width it requires.
- Exact runtime layouts for struct storage buffers in DirectX and OpenGL, and 64-bit integer buffers in native reflection.
- Project options for workgroup access assertions (`[[project.workgroup_access_assertions]]`) and per-entry workgroup size rules (`[project.entry_workgroup_size_rules.<pattern>]`).
- An OpenGL `private_pointer_out_of_bounds_read` policy, `error` by default.

### Improved

- Entry-scoped Metal artifacts drop unreachable helpers, declarations and structs, and target capability checks look only at the code reachable from the selected entries.
- Metal round trips handle typedef chains, bfloat values, constrained templates, unions and `constant` resources.
- HLSL reflection keeps CBV, SRV, UAV and sampler registers apart, and GLSL reflection records `std140` and `std430` layouts.

### Fixed

- Float atomic addition and exchange on Metal buffer elements and fields.
- DirectX lowering of Metal `fabs`, `fmin`, `fmax` and the argument order of `select`. OpenGL lowering of Metal floating-point math and Boolean selection.
- DirectX 16-bit `as_type` reinterpretation, and `gl_SubgroupID` in workgroups with more than one wave.
- OpenGL bitcasts of widened binary16 values.

---

## [3.2.0] - 2026-08-14

### Added

- Native Direct3D and OpenGL loader adapters with artifact verification and SPIR-V specialization.
- Runtime variant dispatch packages with C++17 execution wrappers.
- Deferred native compilation with a versioned request format and an output cache.
- Resumable project translation in isolated worker processes, with checkpoints.
- Several entry artifacts per source, and Metal entry-point discovery.

### Improved

- Metal analysis is faster: scope lookups are indexed and repeated tree walks are gone.
- HLSL compute stages are inferred from `numthreads`.

### Fixed

- OpenGL index narrowing, scalar conversions, Boolean compound assignment and private arrays in specialized callers.
- DirectX aggregate initialization. Metal helper calls keep resource pointer offsets in DirectX and OpenGL output.
- Direct3D shutdown with live pipelines fails instead of continuing.

---

## [3.1.0] - 2026-07-19

### Added

- Runtime execution graphs, variant registries and native loader packages for C11 and C++17 hosts.
- Native compute drivers for Direct3D, headless OpenGL and Vulkan.
- Cooperative-matrix types in CrossGL, lowered to Vulkan KHR cooperative-matrix SPIR-V.
- Pinned MLX project translation to DirectX, OpenGL and Vulkan.

### Improved

- Metal project translation handles templates, overloads, references, pointer inference, constant expressions and wide vectors. DirectX and OpenGL lowering handles fixed arrays, 16-bit arithmetic, storage aliases and specialization constants.

### Fixed

- Arithmetic conversion order, explicit constructors, Boolean compound assignment and conditional expressions.
- Metal alias templates and overload resolution, DirectX entry-point validation and array initialization, OpenGL block-name collisions, and SPIR-V helper returns.

---

## [3.0.0] - 2026-06-23

### Added

- WebGL (GLSL ES) and WebGPU (WGSL) output. Both are output-only.
- DirectX profile aliases `dx11`, `dx12`, `d3d11` and `d3d12`, all producing HLSL.
- Project-scale porting with reports, runtime readiness checks and runtime loader manifests.
- Metal struct template materialization, and lowering of Metal subgroup and atomic operations.
- Runtime parity checks for translated artifacts, naga validation of WGSL output, and SPIR-V assembly and validation.

### Improved

- HLSL compute, GLSL subgroups and specialization constants, SPIR-V storage buffers, and the WGSL, Rust, Mojo and Slang generators cover more of the language.

### Fixed

- Over 100 backend fixes across CUDA, HIP, Metal, GLSL, HLSL, SPIR-V, WGSL, OpenCL, Rust, Mojo and Slang.

---

## [2.0.0] - 2026-06-01

### Breaking Changes

- CrossGL gained generics, traits, enums with data, pattern matching, do-while loops and double-precision types. `.cgl` files written for 2.0 may not parse with 1.x.
- `ShaderNode.stages` is a `StageMap` instead of a `dict`, and AST nodes have `child_nodes()`, `walk()` and `bind_parent_links()`. Code generators expect the new AST.
- Code generators are no longer imported directly. Use `crosstl.translate()` or `crosstl.translator.codegen.registry`.

### Added

- Validation of texture intrinsics, stages and resources.
- Backends register through `BackendSpec` and `SourceSpec`, so third-party backends can plug in.
- Compute, geometry, tessellation, mesh, task and ray-tracing stages on the backends that have them.
- Sphinx documentation with a per-backend API reference.

### Fixed

- Fragment output names in GLSL, vector constructors in the CUDA and HIP parsers, `switch` default ordering, and texture comparison functions.

### Removed

- pytest is no longer a runtime dependency.

---

## [1.1.0] - 2025-12-15

### Added

- Slang, Mojo and Rust backends.
- Backend and source registries.

---

## [1.0.0] - 2025-10-01

### Added

- The CrossGL language, with translation to and from Metal, HLSL, GLSL, SPIR-V, CUDA and HIP for vertex, fragment and compute stages.
