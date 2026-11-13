# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them.

## [Unreleased]

### Fixed

- WGSL output writes HLSL's `f32tof16` and `f16tof32` with `pack2x16float` and `unpack2x16float`: `f32tof16(x)` becomes `pack2x16float(vec2<f32>(x, 0.0))`, which keeps the half in the low 16 bits, and `f16tof32(h)` becomes `unpack2x16float(h).x`, with helpers for vectors. It used to copy both calls into the output, which naga rejects (`no definition in scope for identifier: f32tof16`) (#4407).
- HIP output writes CrossGL's `clockRealtimeEXT()` as `wall_clock64()`, HIP's constant-rate device clock. It used to copy the call into the output, where hipcc rejected the undeclared identifier (#4414).
- A shader-scope constant or constant-buffer member passed as the first argument of a generic overload call, as in `sel(K, x)` or `sel(offset, offset)`, is a value argument. It had been read as the overload's type argument, so Metal, HLSL, Slang, CUDA and HIP output declared `K sel_K_overload0(K a)`, which their compilers reject, and GLSL, SPIR-V and WGSL output refused the program. A leading name now counts as a type argument only when it is not a local or parameter and names a built-in type, a declared structure, enumeration or type alias, or a generic parameter in scope; `cbuffer` and `buffer` block names are values (#3761).
- HLSL output declares a 1D shadow texture, which HLSL input imports from a `Texture1D` or `Texture1DArray` used only with `SampleCmp`, `SampleCmpLevel`, `SampleCmpLevelZero` or `SampleCmpGrad`, as `Texture1D` or `Texture1DArray` again. The round trip used to fail with `ValueError: DirectX texture operation 'textureCompare' requires a declared texture or image resource argument` (#2639, #2640, #2641).
- GLSL output counts the `length()` of a reinterpreted buffer view, such as `(uint16_t*)words` or `(float*)(words + 2)` held in a local, written inline or passed to a helper, in the view's elements: `words.length() * 2` for a `uint16_t` view of `uint` words, less the view's byte and element offsets. It used to call `.length()` on one element read from the buffer, which glslang rejects (`'length' : does not operate on this type`). A view whose element layout is unknown is refused with `OpenGLStoragePointerError` (#3758).

---

## [3.7.0] - 2026-11-13

The [full 3.7.0 notes](https://github.com/CrossGL/crosstl/blob/8283636516c5a1b66c43bcf6c384c7ee5e0610fd/CHANGELOG.md#370---2026-11-13) describe each of the release's 117 changes.

### Breaking Changes

- Every target refuses a function body that reads or writes a name it never declares, with `UndeclaredIdentifierError`, and a call to a function that neither the program nor any language CrossTL reads declares, with `UndeclaredFunctionError`. Both used to be copied into the output.
- GLSL, HLSL and Metal output refuse what they have no spelling for instead of writing it unchanged, such as built-in types and functions of other languages and, in GLSL output, members, semantics and functions nothing declares. GLSL, SPIR-V, HLSL and Metal output refuse an HLSL system value they do not know.
- A storage image format CrossGL does not know is refused with `UnsupportedImageFormatError`.
- Metal and Slang output refuse a constant-buffer member that `@offset` or `@align` places where their declaration would not, with `MemberPlacementError`, and HLSL and GLSL output refuse such placements in the structure of a `ConstantBuffer<T>` that declares no layout. HLSL input refuses a `[[vk::push_constant]]` struct whose DXC layout CrossGL cannot record.
- A helper's `buffer T name[]` parameter takes the form of the buffer its caller passes, and one whose elements have no common stride across std430, HLSL structured buffers and MSL is refused with `StorageBufferParameterUnsupportedError`. HLSL, GLSL, SPIR-V and Slang output used to write an unsized array parameter that their compilers reject.
- The CrossGL output of a Metal, GLSL, HLSL or SPIR-V import starts with `#pragma cgl_rules(...)`, which names the semantic rules of the source language, so the saved CrossGL translates the same as the source.

### Added

- HLSL input translates function and struct templates, including template value parameters, to CrossGL generics, and resolves namespaces by C++ lookup rules.
- CrossGL generic functions take value parameters, as in `generic<T, N>` called as `scale<float, 3>(x)`, and generic structures such as `Box<float>` translate to SPIR-V as well as the other targets.
- Storage images take the packed formats `rgb10_a2`, `rgb10_a2ui` and `r11f_g11f_b10f`.
- Metal input keeps a declaration-only function template that explicit specializations complete.

### Improved

- The Metal importer reads more template-heavy MSL, including member templates selected by a library tag, default template arguments of deduced calls and body-local aliases of dependent types.
- More MLX kernels translate to DirectX and OpenGL. OpenGL output proves index bounds from the comparisons that guard a read, and DirectX and OpenGL output write through 32-bit storage word views.
- Translating large programs to HLSL is faster, with unchanged output.

### Fixed

- 97 fixes, mainly to Metal, Slang, HLSL, SPIR-V and GLSL output and to Metal and HLSL input. Many concern the layout of constant buffers, push-constant blocks and storage buffers, arrays and `inout` arguments passed to functions, matrix order, memory qualifiers, and tessellation and workgroup sizes.

---

## [3.6.0] - 2026-11-08

The [full 3.6.0 notes](https://github.com/CrossGL/crosstl/blob/v3.6.0/CHANGELOG.md#360---2026-11-08) describe each of the release's 238 changes.

### Breaking Changes

- Output refuses a specialization constant it cannot keep specializable, where it used to freeze the constant at its default value.
- The CrossGL parser refuses a token it cannot place, and a malformed `generic<...>` declaration, with a `SyntaxError` that names the construct. It used to skip them.
- The Metal, HLSL and Vulkan GLSL importers refuse a statement or expression they cannot import (`MetalImportRefused`, `DirectXImportRefused`, `SpirvStatementUnsupportedError`) instead of writing a comment in its place.
- `static_assert` is checked for every target: a false condition is refused with `StaticAssertionFailedError`, and one the translator cannot evaluate with `StaticAssertionUnresolvedError`.
- HLSL and Slang input import `SV_VertexID` as `gl_RelativeVertexID`, which leaves out the base vertex as Direct3D does. GLSL, SPIR-V and Metal output read it as `gl_VertexIndex - gl_BaseVertex`.
- HLSL output refuses `volatile` and system-coherent resources, and a call that passes two private pointer views of one local array when either is writable. SPIR-V output refuses `coherent(system)` and `coherent(host)`.

### Added

- A compute, mesh or task entry whose threadgroup size the host chooses at dispatch time, such as a Metal kernel that reads `threads_per_threadgroup`, translates to SPIR-V, GLSL, WGSL, HLSL and Slang. The size comes from specialization constants or WGSL overrides, or from `CROSSGL_WORKGROUP_SIZE_X`, `_Y` and `_Z` macros that the host defines when it compiles the shader.
- HLSL output reads a specialization constant without a default from a `CROSSGL_SPECIALIZE_<name>` macro.
- `@explicit_specialization(primary, types)` replaces the body of a generic function for the given type arguments.
- GLSL and Metal output lower HLSL `ByteAddressBuffer` and `RWByteAddressBuffer`, including signed `InterlockedMin` and `InterlockedMax`.
- SPIR-V output lowers `atomicLoad` and `atomicStore`, and the GLSL importer reads `GL_KHR_memory_scope_semantics` atomics.
- Geometry stages that write several vertex streams translate between GLSL, HLSL and SPIR-V.
- `local_size_x_id`, `local_size_y_id` and `local_size_z_id` make a workgroup dimension a specialization constant on SPIR-V, GLSL, Slang and WGSL output.
- Cooperative matrices gain `cooperative_matrix_diagonal(value)` and independent component types for a multiply-accumulate, and Metal `simdgroup_matrix` kernels that do not use `thread_elements()` lower to GLSL and HLSL.
- The HLSL, Metal, CUDA, HIP, OpenCL and Slang preprocessors define `__COUNTER__`, `__INCLUDE_LEVEL__`, `__BASE_FILE__` and, except for HLSL, `__FILE_NAME__`.

### Improved

- The Metal importer reads more of the template-heavy MSL in MLX, PyTorch and ggml kernels, including constrained templates, `enable_if` overload selection, `decltype` return types and type traits in constraints.
- Importing and translating large Metal sources is faster.
- WGSL output passes array elements, structure members, vector components and module-scope variables to `inout` and `out` parameters through temporaries.
- Slang output lowers the wave shuffle functions that Metal's `simd_shuffle_*` functions import as.
- Refusals that depend only on a program's entry stages or workgroup sizes are reported earlier.

### Fixed

- 184 fixes, mainly to Metal input and to WGSL, Metal, GLSL and HLSL output.

---

## [3.5.0] - 2026-10-30

The [full 3.5.0 notes](https://github.com/CrossGL/crosstl/blob/v3.5.0/CHANGELOG.md#350---2026-10-30) describe each of the release's 291 changes.

### Breaking Changes

- Vertex inputs and varyings take their location from one rule on GLSL, SPIR-V, WGSL and Metal: an explicit location, then the classic semantic table (`POSITION` 0, `TEXCOORD0` 5, `COLOR0` 13 and so on), then the first free location. Host vertex layouts written for the old numbers need updating.
- A storage buffer without `@buffer_layout` holds its elements at std430 offsets on Metal and HLSL, as on the other targets. `buffer T name[]` declares one storage buffer of `T` for every `T`, structures included.
- Metal output samples a combined sampler, such as GLSL `sampler2D`, through a `sampler <texture>Sampler [[sampler(n)]]` argument that the host must bind. It used to sample through a fixed inline sampler.
- CrossGL refuses non-square `floatNxM` and `doubleNxM` spellings (`AmbiguousMatrixSpellingError`), undeclared type names, and `gl_` attributes that name no built-in or are used where GLSL does not declare them. `(a, b)` is the C comma operator. `sizeof` folds to a constant on Metal, HLSL and GLSL and is refused for operands it cannot size.
- CUDA and HIP output refuse a construct they cannot lower with `CudaLoweringRefused` or `HipLoweringRefused`. They used to write a comment and a placeholder value in its place.
- Target limits are refused with typed errors: derivatives in compute, mesh, task and ray generation stages on Metal, 64-bit integers, doubles and bfloat16 on WGSL, constant buffers over 64 KiB on HLSL, and `WaveSize`, mesh, amplification and ray-tracing stages on the `dx10` and `dx11` profiles.
- Importers refuse constructs they cannot translate faithfully: Metal `sinpi` and its family, ray-tracing functions, indirect command buffers and rasterization rate maps, GLSL `tensorARM` uniforms, sampler feedback textures and OpenCL C device-side enqueue events.
- Mojo output declares constants with `comptime` instead of `alias`, and refuses a vector where a scalar is expected.

### Added

- `crosstl doctor` lists, for each backend, the validation tools CrossTL looks for and which of them are on `PATH`.
- GLSL mesh and task shaders (`GL_EXT_mesh_shader`) translate to HLSL and Metal, and Metal `[[mesh]]` and `[[object]]` functions import as mesh and task stages.
- CrossGL structures may declare member functions, and CrossGL, GLSL, HLSL, Slang and CUDA input accept a structure declared in a function body.
- Sparse texture residency queries translate to GLSL, Metal, HLSL and SPIR-V, and the GLSL importer reads `GL_ARB_sparse_texture2` and `GL_ARB_sparse_texture_clamp`.
- `gl_InstanceIndex` translates to OpenGL, OpenGL ES, HLSL and SPIR-V. Metal `constexpr sampler` state translates to OpenGL through host sampler objects.
- `metal_sizeof(x)` measures a type with Metal's layout rules. The SPIR-V importer reads subpass inputs.
- GLSL output reads a specialization constant without a default from a `CROSSGL_SPECIALIZE_<name>` macro.
- Project translation for DirectX and OpenGL flattens a compute workgroup whose shape the target cannot declare but whose thread count it can, and a project can declare read-only entry storage immutable with `immutable_storage`.

### Improved

- The Metal importer reads more of the MSL in real headers and MLX kernels: template argument deduction, partial specializations, `decltype`, `conditional_t` and the `<simd/simd.h>` aliases.
- WGSL output monomorphizes generic functions and passes `inout` and `out` parameters as pointers.
- Inline ray queries have one CrossGL spelling, the DXR method names, and translate between HLSL, GLSL and SPIR-V.
- Metal argument table indices become SPIR-V bindings, and fixed sampler state becomes host samplers on Direct3D and Vulkan.
- SPIR-V output folds module-scope constant expressions exactly in binary32, HLSL and GLSL output write the float limits such as `FLT_MAX` exactly, and Metal `half` matrices keep binary16 values on SPIR-V and WGSL.
- Arrays of HLSL structured buffers keep their tight element layout on GLSL, SPIR-V and Metal.
- Translating large sources is faster.

### Fixed

- 226 fixes, mainly to CUDA, HIP, WGSL, Metal, HLSL and Slang output.

---

## [3.4.0] - 2026-10-23

### Breaking Changes

- CrossTL requires Python 3.11 or later. Python 3.8, 3.9 and 3.10 have reached end of life and are no longer supported.
- CrossGL syntax can change between releases until CrossGL expresses everything its targets can. In this release, debug prints are `printf(format, values...)` instead of `debugPrintfEXT`, comparison samplers have their own `comparison_sampler` type, and HLSL and Metal textures import as separate textures and samplers.
- A Metal, CUDA, HIP or OpenCL kernel whose results depend on a threadgroup size that the host picks at dispatch is refused for HLSL, GLSL, SPIR-V and WGSL output, which used to fix the size at 1, 1, 1. Pass the dispatch size as `source_options={"threadgroup_size": (x, y, z)}` or in a project workgroup-size rule.
- A CrossGL function body that does not parse is refused; it used to become an empty block. An untyped `var` whose type cannot be inferred from its initializer, or that has no initializer, is refused instead of being declared `float`.
- CrossGL `matCxR` has GLSL's column-by-row meaning on every target, and HLSL `floatRxC` imports as `matCxR`. Comparison operators and `!` are componentwise on vectors, as in HLSL, MSL, WGSL and SPIR-V. Ray payload access qualifiers are `@ray_payload_read(...)` and `@ray_payload_write(...)`.
- The Metal, OpenGL, WebGL, DirectX, SPIR-V, Slang and Rust generators, and the HLSL and Metal importers, refuse a construct they cannot translate with a typed error that lists every such construct, such as `MetalLoweringRefused` or `SpirvLoweringRefused`. They used to write an `/* unsupported */` or `; WARNING` comment next to a neutral value and return a program that ran differently.
- SPIR-V output from HLSL takes descriptor sets and bindings from `register(Xn, spaceM)` and `[[vk::binding]]` instead of numbering resources in declaration order, and refuses two resources on one register. OpenGL output gives each descriptor set or register space small per-class binding points instead of `set * 1024`.

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
- HLSL geometry, hull and domain stages translate to GLSL and SPIR-V. Metal, which has no geometry or tessellation stages, refuses them with a typed error.
- GLSL subgroup built-ins, including the prefix, clustered and ballot forms, translate to SPIR-V, HLSL and Metal.
- Source arithmetic profiles for binary32 addition, subtraction, multiplication, division, remainder, precise arctangent and logarithms. DirectX, OpenGL and Metal output get native controls for rounding, subnormals, signed zeros and non-finite values.
- Thread-grid dispatch regions, fixed partial workgroups, Boolean subgroup votes and software subgroup products. Unsupported collective control flow is reported as an error.
- Native runtime support for Boolean, binary16, bfloat16, byte and 64-bit integer storage, shared allocation views and immutable DirectX constants.
- Checked integer metadata preconditions carried through project artifacts, reflection and runtime packages.
- MLX integration demos for array creation, copies, casts, reductions, indexing, scatter and random generation through translated runtime packages.
- Package metadata lives in `pyproject.toml` with a pinned Flit build backend. Releases install the wheel and the source distribution in isolation and verify the published PyPI archives before the GitHub release is created.

### Improved

- Every supported Python version, 3.11 through 3.15, is tested.
- Unsized HLSL resource arrays get their size on Metal and OpenGL from bounded dynamic indexes as well as constant ones, so translated shaders no longer read past the array.
- HLSL texture samples return the texture's element type, and storage-image loads and stores use four-component texels as GLSL defines them.
- CUDA and HIP buffer elements such as `float3` and structs with vector members keep their host layout on every target.
- SPIR-V modules for mesh, task and ray-tracing stages, which need SPIR-V 1.4, declare storage buffers in the `StorageBuffer` storage class and pass `spirv-val`.
- GLSL stage inputs and outputs are named so the stages link, and never redeclare a global.
- Metal import and graphics code generation are faster on large kernels.
- Buffer layouts carry across targets: HLSL structured, append and consume buffers, Metal packed and natural layouts, and GLSL std140, std430 and scalar blocks keep their offsets. HLSL `row_major`, `packoffset` and `pack_matrix` are honoured, and `GetDimensions` reports the element stride on GLSL, SPIR-V, CUDA and HIP.
- Texture sampling keeps its meaning across targets, including LOD clamps, implicit LOD outside fragment stages, projected depth comparisons, Metal array layers and explicit sampler pairs.
- OpenGL output declares the lowest GLSL or GLSL ES `#version` that provides every construct the program uses.
- Stage interfaces translate more faithfully: GLSL interface blocks keep their member locations, struct-typed varyings are flattened for SPIR-V and split for Metal, a GLSL fragment shader's several outputs become one HLSL or MSL structure, and each SPIR-V entry point has its own interface locations.
- `SV_ClipDistanceN` and `SV_CullDistanceN` keep their index, and GLSL and SPIR-V pack them into one distance array in DXC's order.
- Arrays of uniform blocks bind one buffer per element, Metal `constant` matrix parameters bind as HLSL constant buffers, and SPIR-V bindings of Metal resource parameters follow MSL's separate buffer, texture and sampler indices.
- The SPIR-V importer rebuilds structured control flow, keeps integer signedness, extended arithmetic and `NoContraction`, and imports arrays of storage blocks.
- The Metal importer handles more real-world sources: `__METAL_VERSION__` and `__METAL__` are predefined, shared headers that use `<simd/simd.h>` types translate without Apple's SDK headers, and kernels with 16-bit thread-index attributes, `unsigned int` template arguments, `::metal::`-qualified SFINAE constraints or type traits in explicit instantiations translate.
- A function whose recursive calls are all tail calls is translated as a loop instead of being refused.
- Native Metal dispatches reuse a long-lived worker process instead of starting one per dispatch.
- Precise Metal square root, logarithm, power and `atan2` keep signed zeros, special values and underflow behaviour under explicit operand profiles.
- HLSL, GLSL and Metal output keep integer promotions, vector initialization, byte narrowing, binary16 selection, bfloat rounding, atomic destinations and argument evaluation order.
- Source maps no longer guess line locations. Only artifacts that preserve lines carry line mappings.
- MLX project files moved to `demos/integrations/mlx`.

### Fixed

- The parser no longer hangs on an unterminated shader declaration, and parses `if`, `else` and loop bodies without braces.
- HLSL output renames GLSL and Metal identifiers that HLSL reads as keywords, such as `line`, `point`, `vector` and `Buffer`.
- Metal SIMD-group and quad-group functions declared behind `METAL_MAYBE_UNDEF` are recognized.
- SPIR-V output keeps the source semantics of signed remainder, `fmod`, signed right shifts, mixed-sign arithmetic and NaN tests, copies locals initialised from buffer elements, and passes loose uniforms in a `CrossGLDefaultUniforms` block instead of private variables the host cannot set.
- HLSL output stores the written components of a GLSL buffer-block vector; it used to assign to a load, which DXC rejects.
- HLSL output declares vertex, instance, primitive and sample indices as `uint`, as DXIL requires.
- GLSL output rebuilds a flattened stage-input structure used as a whole value, and guarded multi-stage GLSL declares one `main` per stage.
- Functions that share a name with a module-scope value are renamed in HLSL, GLSL and MSL output, and constants that read struct constants are declared after the structs.
- HLSL `SubpassInput` keeps its type through CrossGL, and `gl_PointCoord` reaches Metal as `[[point_coord]]`. OpenGL refuses input attachments and HLSL refuses `gl_PointCoord`.
- Metal sparse texture reads, colour-attachment inputs on targets without framebuffer reads and array literals passed to Metal array parameters are refused with a typed error instead of producing code that does not compile. WebGL refusals are typed as well.
- Metal colour-attachment outputs can be any scalar or vector of up to four components, and casts after `return` import correctly.
- GLSL mesh built-in arrays and named `case` labels translate to SPIR-V.
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
