# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them.

## [Unreleased]

### Breaking Changes

- A type name that is neither built in nor declared by the program, as a structure, enumeration, alias, constant block or generic parameter, is refused with a typed error. Every generator used to copy it into the output.
- WGSL output refuses 64-bit integers, doubles and bfloat16, the vectors and matrices built on them and conversions such as `double(x)`, with `WGSLNumericTypeUnsupportedError`. It used to narrow `double` to `f32` and `long` to `i32`, which changed host buffer layouts, or copy the type unchanged.
- `buffer T name[]` and `buffer T name[N]` declare one storage buffer of `T` elements for every `T`, structures included, as Metal's `device T*` does. GLSL buffer block arrays (`buffer B { ... } name[N];`) keep their meaning as arrays of descriptors.
- CrossGL `sizeof` of a scalar, a 2- or 4-component vector or a sized array is folded to a constant on Metal, HLSL and GLSL. Other operands are refused with `SizeofUnsupportedError`; `sizeof` used to be written out unchanged, so GLSL rejected it and HLSL and Metal measured with their own layout rules.
- The Metal importer refuses `sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi` and `atan2pi`, which are exact at integers and half-integers where `sin(x * pi)` is not, and calls to functions that the source declares but never defines. Both used to reach every target as calls to undeclared functions.
- Metal indirect command buffers (`command_buffer`, `render_command`, `compute_command`) and rasterization rate maps are refused with `MetalRuntimeTypeUnsupportedError` for every target other than Metal when a stage uses them. HLSL and GLSL output used to copy the type names.
- HLSL output refuses a constant buffer larger than Direct3D's 64 KiB limit, such as a large Metal `constant T&` argument or GLSL uniform block, with `ConstantBufferTooLargeError`, and matrix varyings whose semantic indices overlap with `InterfaceLocationConflictError`. DXC used to reject the generated shader.
- The GLSL importer refuses `tensorARM` uniforms with `GLSLTypeUnsupportedError`. It used to drop them, so every target referenced an undeclared name. GLSL output refuses string literals outside `#include`, `#line` and `debugPrintfEXT` formats.
- The Metal importer refuses ray-tracing functions, and the uses of mesh and object functions it cannot import, such as `set_indices`, with `MetalImportRefused`. They used to be printed as plain helpers that still called Metal methods.

### Added

- `crosstl doctor` lists, for each backend, the validation tools CrossTL looks for and which of them are on `PATH`. A validation that cannot run now names the missing tool.
- CrossGL `textureResidency(read, texel)` and `textureResident(code)` query sparse texture residency on GLSL, Metal, HLSL and SPIR-V.
- `metal_sizeof(x)` measures a type with Metal's layout rules. The Metal importer writes it for operands it cannot size, such as template parameters.
- Metal `[[mesh]]` and `[[object]]` functions import as CrossGL mesh and task stages.
- `gl_InstanceIndex` translates to OpenGL as `gl_InstanceID + gl_BaseInstance`, to OpenGL ES as `gl_InstanceID`, to HLSL as `SV_InstanceID` and to SPIR-V. As with `gl_VertexIndex`, HLSL's value does not include the base instance.
- Metal fixed sampler state (`constexpr sampler`) translates to OpenGL through host sampler objects. GLSL output starts with a `// crosstl-fixed-samplers:` line that lists each state with its uniforms and texture units. WebGL still refuses it.
- The SPIR-V importer reads subpass inputs as `subpassLoad`. MSL `clz` and `ctz` import through `findMSB` and `findLSB` for 8-, 16- and 32-bit operands, and Metal output lowers `findMSB` and `findLSB` instead of copying them.

### Improved

- WGSL output monomorphizes generic functions, passes `inout` and `out` parameters as pointers, copies written parameters into locals, binds every entry parameter with `@location` or `@builtin` and gives integer varyings `@interpolate(flat)`. Memory barriers, matrix-scalar `+`, `-` and `/`, `matrixCompMult`, `outerProduct`, `dot4add_i8packed` and `dot4add_u8packed` are lowered to WGSL instead of being copied.
- Inline ray queries have one CrossGL spelling, the DXR method names, and translate between HLSL, GLSL and SPIR-V.
- The Metal importer reads more of the MSL that real headers use: function templates deduce their arguments before applying defaults, partial specializations constrained with `enable_if` are selected, alias templates work as constructors, a local alias may reuse its template's name, `simd` type aliases work as constructors, an unscoped enumerator is found in the enclosing scope as in C++, and `powr` and `exp10` are lowered for other targets.
- Metal sources keep their argument table indices as SPIR-V bindings, and their fixed sampler state becomes host samplers on Direct3D and Vulkan. Metal output binds arrays of HLSL structured buffers and GLSL storage blocks as one entry argument per element, as MSL requires. Vulkan subpass inputs become texture reads on targets without input attachments.
- Generic functions deduce a type parameter from a vector argument, as in `pick(ivec2(1, 2))` for `T pick<T>(vec2<T> v)`, and generically spelled vectors such as `vec2<u16>` map to the right type on GLSL, HLSL and MSL.
- The HLSL importer types shader-scope globals and keeps their initializers, and imports `[[vk::shader_record_ext]]` constant buffers as shader record blocks. The Slang importer keeps `groupshared` on globals.
- Translating large sources is faster: the lexers pick token patterns by first character, and the Metal preprocessor and several lowering passes walk the program fewer times.
- Metal output renames identifiers that MSL reads as keywords, such as `kernel`, `device`, `thread` and `half`, gives `gl_InstanceIndex` an `[[instance_id]]` input, binds a ray generation stage's launch index and size to the kernel grid and spells its ray flags as constants, and passes resources and module-scope groupshared values to mesh and object functions.
- The Metal importer types matrix arithmetic, the `saturate` family and the results of texture methods, imports depth texture reads as the depth value, and keeps `[[attribute(n)]]` vertex inputs at location `n` on every target instead of renaming them to HLSL semantics.
- SPIR-V output folds module-scope constant expressions exactly, rounding binary32 results after each operation, instead of emitting instructions that module scope cannot hold. Shading-rate built-ins, per-primitive mesh outputs and arrays of constant buffers are declared with their SPIR-V built-ins and capabilities.
- `INFINITY`, `NAN`, `FLT_MAX`, `FLT_MIN`, `FLT_EPSILON` and the other standard float limits are written as exact binary32 values on HLSL and GLSL.
- Arrays of HLSL structured buffers keep HLSL's tight element layout on GLSL, SPIR-V and Metal.
- Texel lookup offsets that a program keeps in constant locals or computes in short constant loops become constant expressions on GLSL, HLSL, WGSL, SPIR-V and Slang. SPIR-V output refuses a non-constant offset outside gathers instead of writing an invalid module.
- Metal `half` matrices translate to SPIR-V and WGSL as float matrices that hold binary16 values, rounded after each operation, and `half` operands in mixed expressions are typed. The Metal importer maps the `simd::` vector and matrix types like the Metal ones.
- GLSL `length()` of a vector or matrix becomes its component or column count on other targets.
- Structure member extents named by constants, such as `float data[Total];` with `constant constexpr size_t Total = X * Y * Z;` declared after the structure, are folded to integers on every target.
- The Metal importer resolves the `<simd/simd.h>` aliases such as `simd::float4`, `simd_float4x4` and `vector_float3`, declares the `<metal_stdlib>` plain-data structures a source uses without declaring, such as the tessellation factor and indirect argument records, and imports `atomic_fetch_sub_explicit` as an atomic add of the negated value instead of copying the call.
- HLSL implicit vector truncation in assignments and initializers becomes a leading-component swizzle on every target.

### Fixed

- Mojo vector comparisons produce boolean masks with element-wise comparison methods. Boolean vector reductions preserve only the logical lanes of padded three-component vectors.

- WGSL output gives resources without an explicit binding the lowest free slots in their bind group, as GLSL and HLSL output do. Explicit bindings in any group used to push them higher: next to `cbuffer Params @binding(3)`, two storage buffers without a binding took slots 4 and 5 instead of 0 and 1.
- Metal output of matrix shape conversions such as `(float3x3)m`, of `as_type<long>` and `as_type<ulong>`, and of fragment colour attachment inputs compiles, and HLSL `GetDimensions` assigns its stride output on Metal.
- HLSL output of Metal sources gives `half` fragment returns `SV_Target`, declares `half` textures without a stray array size and drops `<TargetConditionals.h>`.
- An early `return;` in a GLSL `main` that writes stage outputs returns those outputs on every target.
- CUDA output writes CrossGL scalar casts such as `f64(x)` with C type names.
- CUDA output passes the address of the target to block- and system-scoped atomics such as `atomicAdd_block` and `atomicInc_system`, as it already did for `atomicAdd` and `atomicInc`.
- A program with several stages of one kind, such as a Metal library with several vertex functions, has every stage rewritten, not only the first.
- Translating a malformed shader no longer takes time that grows much faster than its length.
- An assignment used as a value keeps its own parentheses: `int f = (e = 4) + 1;` used to become `e = 4 + 1` on several targets, which assigns 5.
- SPIR-V output short-circuits `&&`, `||` and `?:`. The right operand and both arms used to run unconditionally.
- GLSL output declares `long`, `ulong`, `size_t` and `ptrdiff_t` as 64-bit integers instead of 32-bit ones, which changed arithmetic and block layouts; GLSL ES and WebGL refuse them. SPIR-V output converts `size_t(x)` and `ptrdiff_t(x)` and widens `uint` without sign extension.
- 8- and 16-bit integer vector arithmetic wraps at its own width on GLSL and Metal, and Metal widens native `ushort2` and similar fields and buffer elements when it reads them.
- Storage buffer arrays translate to GLSL, SPIR-V, HLSL, WGSL, Slang and Metal. Sized arrays on HLSL used to be `static` arrays whose stores never reached memory, and unsized arrays of structures were refused on HLSL and WGSL.
- A GLSL entry point that writes one of its inputs, built-ins included, writes a local copy instead of the read-only input.
- HLSL entry points that write stage outputs through `out` and `inout` parameters translate to GLSL, HLSL, Metal, SPIR-V, WGSL and Slang, and `SV_Target<N>` reaches colour attachment N on GLSL, WGSL, Slang and Metal. Multi-stage shaders written this way used to fail, and GLSL and WGSL numbered attachments in declaration order.
- WGSL output of a `switch` has one `default` arm and keeps fallthrough by merging case bodies, writes `discard;`, and converts scalar operands of vector stores, shifts and bitwise operators, which naga used to reject.
- An HLSL vertex input with `SV_Position` is a regular vertex attribute on GLSL, Metal, WGSL and SPIR-V. Metal and WGSL rejected it, and SPIR-V read the `gl_Position` output instead.
- Function-scope `static` variables keep their value across calls on every target, and HLSL output keeps the initializers of static globals.
- Vector `&&` and `||` combine per component on every target. HLSL 2021 and WGSL output used to be invalid, and GLSL refused it.
- Metal output of HLSL vertex inputs that mix attributes with built-ins such as `SV_VertexID` compiles, and an `int` `SV_VertexID` is accepted.
- Metal fragment parameters whose members carry `[[color(n)]]` read colour attachments on every target, or are refused where the target has no framebuffer fetch. They used to become ordinary inputs. A default-constructed Metal `sampler` gets MSL's default sampler state.
- The Metal importer reads `m.columns[i]` of a matrix as the column `m[i]`. The `columns` array used as a value is refused with `MetalMatrixColumnsError`.
- HLSL output stores bfloat16 vectors as per-lane `uint` payloads, writes `fwidthFine` and `fwidthCoarse` with fine and coarse derivatives, and the HLSL importer maps packed 8-bit structured buffer elements.
- A scalar swizzle such as `s.xxx` becomes a vector constructor on Metal, WGSL and GLSL ES, which do not accept it.
- Slang output reads `gl_NumWorkGroups` as `WorkgroupCount()`, and Metal output infers the type of a shader-scope `let` from its initializer.
- Metal `device T&` and `const device T&` arguments that reference a structure read the buffer on every target. HLSL, GLSL and SPIR-V used to declare them as stage inputs, so the shader read vertex attributes or nothing.
- Metal `[[vertex_id]]`, `[[instance_id]]` and `[[primitive_id]]` arguments keep their `uint` type, so arithmetic on them wraps as in MSL. They used to import as `int`.
- A GLSL local whose initializer reads an outer variable of the same name, as in `float scale = scale * 3.0;`, is renamed on HLSL, Metal, CUDA, HIP and Slang, where the initializer would read the new, uninitialized variable. The GLSL importer no longer reads a local that shadows a member of an anonymous buffer block, or of a uniform block it flattens, as that member.
- GLSL output zero-initializes arrays with one constructor argument per element, and refuses an array whose size it cannot evaluate with `OpenGLZeroValueError`. HLSL output writes a one-argument vector constructor of unknown width as a cast, which splats a scalar.
- Locals named after built-ins that GLSL and HLSL output call, such as `texture`, `textureSize`, `mod` and `imageSize`, are renamed so they no longer hide those functions.
- HLSL output adds no second `SV_Position` parameter to a fragment entry whose input structure has one, declares `int` members that carry `SV_RenderTargetArrayIndex`, `SV_ViewportArrayIndex`, `SV_PrimitiveID` or an index system value as `uint`, and gives members named `view`, `layer` or `viewport` no system-value semantic. A Metal varying `float3 view` used to carry the view index.
- Early fragment tests reach HLSL as `[earlydepthstencil]`, SPIR-V and Metal output, and the HLSL importer reads `[earlydepthstencil]`. Only GLSL output kept them.
- The HLSL importer keeps a value-returning function marked `[numthreads]` as a helper, as DXC does, and GLSL output converts a scalar returned by a `float4` pixel entry to the output's type.
- SPIR-V output declares immutable scalar, vector and matrix globals with constant initializers as constants, so `constant uint T = W * 3;` is valid, and flattens structure members of vertex outputs and fragment inputs that carry no location, such as the nested structures of a Metal mesh pipeline's `[[stage_in]]`. They used to overlap the next member's location.
- GLSL output of a multi-stage file emits each helper only under the macros of the stages that call it, so a fragment-only helper that calls `fwidth` no longer breaks the vertex stage. Written half-precision stage inputs are copied into locals, and varyings no longer take the names of entry resource parameters.
- Depth comparisons with 16-bit float reference values translate.

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
