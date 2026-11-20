# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them. Entries for newer changes wait in `changelog.d/` until the next release is prepared.

## [Unreleased]

---

## [3.8.0] - 2026-11-20

### Breaking Changes

- Every refusal the translator and the importers raise has its own diagnostic class, and the last places that wrote an unsupported construct through, or put a comment, a zero or an empty block in its place, refuse it instead. Name lookups in every generator and importer use exact tables, so an unknown stage (`UnknownShaderStageError`), semantic, built-in, operator or type is refused rather than written as it was. CUDA and HIP output raise `CudaLoweringRefused` and `HipLoweringRefused` where they used to write `/* unsupported ... */` with a fallback value, and the CrossGL parser refuses a function body it cannot read instead of reading it as an empty block (#4471, #4496).
- CUDA and HIP input refuse directed-rounding intrinsics such as `__fadd_rd`, `__fmul_ru` and `__int2float_rz` with `DirectedRoundingUnsupportedError`; they used to translate as the operation that rounds to nearest. The `_rn` intrinsics and the `_rd`, `_ru` and `_rz` conversions from float to integer still translate. Device-side `cudaMalloc`, `cudaMemset`, `cudaFree`, `cudaMemcpy*`, graph calls and kernel launches, and their HIP forms, are refused with typed diagnostics; they used to become comments and `cudaSuccess` or 0 (#4501, #4595).
- CUDA and HIP output refuse vertex, fragment, geometry, tessellation, mesh, task and ray-tracing entries with `StageUnsupportedError`, since a `__global__` kernel is their only entry point. Such an entry used to become a `__device__` function, so the output had no entry point, and a geometry stream discarded every vertex (#4592).
- GLSL and GLSL ES output refuse a program with two entries of one stage with `OpenGLRepeatedStageEntryError`, since a GLSL compilation unit has one entry per stage. Whole-file output used to keep one of two kernels as `main` and turn the other into an ordinary function, or define `main` twice. Translate each entry with `entry_point=` or `crosstl.translate_entries` instead (#4594).
- WGSL, Rust and Mojo output refuse a function that must be evaluated without contraction (`@precise`) with `PreciseEvaluationUnsupportedError`, since none of them can forbid fused multiply-adds, and HIP output refuses matrix and half arithmetic in one. Mojo output refuses wave operations, for which it has no subgroup primitive (#4530, #4537).
- Metal input expands `STEEL_CONST`, `C10_METAL_CONSTEXPR`, `mittens_DEFAULT_ALIGN` and `c10::metal::array` from the project headers that define them instead of recognizing the names, so a source that uses them needs those headers on its include path. Without a definition they are now a syntax error. `mittens_DEFAULT_ALIGN` used to be skipped, which dropped its `alignas(16)` (#4678).

### Added

- `crosstl.translate_entries(file_path, backend, entry_points)` translates each named entry point to its own artifact from one import of the source, as `translate(..., entry_point=...)` does for one. Project translation splits a unit with two entries of one stage into one artifact per entry for GLSL and GLSL ES unless the project sizes the whole unit, and an `entry_workgroup_size_rules` table for the unit no longer turns that split off (#4594, #4692).

### Improved

- Translation takes less CPU time on large programs, Metal ones most: on one machine, MLX's `scan.metal` to Metal took 239 s instead of 355 s. The output is byte-identical (#4493, #4499, #4634).
- The Metal importer reads more template-heavy MSL: static member calls through member aliases, such as `Tile<...>::Frag_t::mma(...)`, and overloads that differ only in where a tag parameter sits; `auto&` reference aliases bound through a local or parameter receiver; member alias templates named through a function-local owner alias; integral-constant arithmetic over block-scope constants, member aliases and `const_for_loop` indices; generic-lambda callbacks of Boolean dispatch helpers, as in `dispatch_bool(gid + 1 >= n, [&](auto kSafe) { ... })`, which become the callback's two instantiations so that `kSafe.value` is a constant expression; and calls to concrete member overloads that take references, such as `add(thread float2& d, ...)` with a local `float2`, which used to be refused as ambiguous (#4645, #4649, #4660, #4672, #4632, #4765).
- The generators, parsers and importers are split into smaller modules by concern, and their longest functions into one method per case, with unchanged output and diagnostics. Every module path of 3.7.0 except the internal `crosstl.translator.codegen.ast_utils` still imports (#4641, #4642, #4644, #4647, #4648, #4650, #4652–#4656, #4662–#4664, #4676, #4677, #4679, #4682, #4683, #4686–#4688, #4693–#4695, #4700, #4706, #4712, #4713, #4716, #4717, #4725, #4727, #4729, #4742, #4761–#4763).
- GLSL output's software subgroups accept a loop or helper that leaves on a subgroup reduction of a value every lane computes alike, since in a workgroup of whole subgroups every invocation leaves together, as HLSL output already does. A helper call that such a condition may skip is still refused (#4470).

### Fixed

- Slang and HLSL output convert a product or an element index applied directly to a ray-tracing transform call (`ObjectToWorld3x4() * p`, `p * WorldToObject4x3()`, `ObjectToWorld3x4()[2][1]`) the same way as on a local of the call's type, as `mul(...)` and a swapped index. The Slang generator did not know the return types of `ObjectToWorld`, `ObjectToWorld3x4`, `ObjectToWorld4x3`, `WorldToObject`, `WorldToObject3x4` and `WorldToObject4x3`, and the HLSL generator did not know them for the unsuffixed `ObjectToWorld` and `WorldToObject`, so slangc and DXC rejected the product and Slang read the transposed element (#4558).
- A zero-parameter function the program defines under the name of a DXR intrinsic, such as `vec3 WorldRayOrigin()` or `float RayTCurrent()`, is called as the program's own function in ray-tracing stages. OpenGL and Vulkan output used to replace the call with the GL_EXT_ray_tracing built-in (`gl_WorldRayOriginEXT`, `gl_RayTmaxEXT`, ...), so the shader read the ray state instead, and Metal output refused it as a ray-state read. HLSL output renames such a function (`ObjectToWorld_`), since DXC reads a call to it as the intrinsic when the return types differ (#4571). An overload that takes arguments, such as `vec3 WorldRayOrigin(float k)`, does not hide the intrinsic: `WorldRayOrigin()` next to it still reads the built-in.
- HIP and OpenCL C input read a function-local `struct Inner a;` as a use of the earlier `Inner`. It used to produce an empty `probe_Inner` record, so the `a.x` that followed did not compile in Metal output (#4469).
- HIP and OpenCL C input refuse a record that is declared without a body, never completed and used by value, as in `struct Never; struct Outer { struct Never n; };` or a local `struct Never v;`, with an `incomplete record type` error that names the record. The same goes for a by-value use that comes before the record's body, for a by-value parameter of a function that has a body, and for a record that holds itself by value, as in `struct Node { struct Node n; };`. A pointer to it, a prototype parameter, and a use after the body are still accepted (#4469).
- HIP and OpenCL C input accept a structure member written with the `struct` keyword, as in `struct Outer { struct Inner inner; int k; };`. The member names the earlier `Inner` (C11 6.7.2.3); it declares no record. It used to be read as a second, body-less `Inner` that collided with the first and stopped the translation with `ValueError: Conflicting HIP nested record output name 'Inner'` (#4469). The same holds for `union` and `class` members and for `struct Inner a, b;` and `struct Inner* p;` declarators.
- HLSL input imports matrices returned by the ray-tracing transform intrinsics (`ObjectToWorld`, `ObjectToWorld3x4`, `ObjectToWorld4x3`, `WorldToObject`, `WorldToObject3x4` and `WorldToObject4x3`), `RayQuery` intersection transform methods and `dx::HitObject` transform methods with their `float3x4` and `float4x3` shapes. Storing one of those calls to a matrix variable used to be refused as a matrix conversion from a value of unknown type (#4542).
- HLSL input imports a scalar stored to a matrix as HLSL reads it, every component set to the scalar: `float3x3 m = 1.0;`, `float3x3 q = x;`, `m = 2.0;`, `return s;` from a matrix function, a scalar passed to a matrix parameter and scalar expressions such as `x * 2.0 + 1.0`, `c ? x : 1.0` or `(float)i` now become `mat3(vec3(1.0), vec3(1.0), vec3(1.0))` and so on. They used to be written as `mat3(1.0)`, which is the identity matrix (#3652).
- Metal `precise::pow` keeps its 16 ulp bound on every target. Its finite path called the target's own `pow`, whose accuracy GLSL, HLSL and SPIR-V inherit from `exp2(y * log2(x))` and which Metal's fast-math mode replaces with `fast::pow`. Results between 2^-120 and 2^120 were off by up to 483 ulp, and results near 2^128 or 2^-126 overflowed or vanished. The finite path now runs the fdlibm binary32 power over every normal result and overflow; only subnormal results and subnormal operands keep the target's `pow`, since flushing them is the target's choice. An unqualified `metal::pow` is unchanged. The new conformance case `m_precise_pow` is within 1 ulp on Metal, OpenGL (llvmpipe) and Vulkan (lavapipe, MoltenVK, KosmicKrisp) under strict and relaxed compilation.
- `mix(x, y, a)` whose selector is a call to `any`, `all`, `isnan` or `isinf` now selects on Metal, HLSL, Slang, WGSL, CUDA and HIP instead of blending, so an unselected infinite or NaN operand no longer turns the result into NaN, and naga accepts the WGSL output. The generators gave these calls no result type, so the boolean `mix` lowering did not see a `bool` or `bvecN` selector. CUDA and HIP cover the scalar forms (#4538).
- CrossGL input reports a typed `SyntaxError` instead of raising `TypeError` when a range expression appears in a type-argument list, while keeping single-index and value-position ranges intact, and the Mojo importer now rejects malformed `..` dots in generic and type arguments (#4502).
- SPIR-V output reads `gl_InstanceID` (and HLSL's `SV_InstanceID`) as `InstanceIndex - BaseInstance` with the `DrawParameters` capability, as slangc and DXC do, so it counts from zero in draws with a non-zero first instance like every other target. It used to decorate it as a bare `InstanceIndex`. The SPIR-V importer now reads `BuiltIn InstanceIndex` as `gl_InstanceIndex`, so a SPIR-V round trip keeps the bare built-in (#4506).
- Every target except CrossGL, CUDA, HIP and WGSL binds a `var<storage>` variable that is not a runtime-sized array as a storage buffer: a module-scope fixed-size array, scalar, vector or matrix becomes a buffer (`x[0]` for a single value), a structure a std430 buffer block, and an entry parameter a runtime-sized buffer parameter. HLSL used to place such a variable in the read-only `$Globals` constant buffer and GLSL in a `uniform`, Metal and WGSL copied a fixed-size array parameter into a private array, and CUDA, HIP, Mojo and Rust passed a scalar parameter by value, so stores never reached memory. A module-scope `var<uniform>` becomes a constant block with its group and binding, and one with an initializer becomes a program constant; HLSL and GLSL used to drop the binding, and Metal and HLSL dropped the initializer of an OpenCL `__constant` table. CUDA and HIP input import a namespace-scope `__device__` or `__managed__` variable as a storage buffer and refuse one with an initializer; they used to import it as a per-invocation private variable.
- HIP and OpenCL input report a `SyntaxError` when a template header has no declaration after it, instead of raising `AttributeError`.
- WGSL output converts an `int` branch of a conditional to the other branch's `uint` or `float` type, scalar or vector, as GLSL does. It used to pass both branches to `select()` unconverted, which naga rejects (#4512).
- WGSL output converts an `int` or `uint` argument of a float built-in such as `max(f, k)`, `step(k, f)`, `mix`, `pow`, `atan(y, x)`, `mod` or `fma` to `f32`, as GLSL's overload resolution does, and replicates it beside a vector. Float-only built-ins such as `sqrt(k)`, `sin(u)`, `length(iv)`, `normalize(iv)` and `dot(v, iv)` convert it even when no argument is floating point. It used to pass the integer as it was, which naga rejected (#4503).
- WGSL output resolves scalar component-wise built-ins using GLSL's floating-point-only overloads and rejects calls whose argument types cannot be inferred, instead of emitting invalid typed calls (#4508).
- WGSL output converts an `int` or `ivecN` operand of `&`, `|` and `^`, and of vector `min`, `max` and `clamp`, to the unsigned type when another operand is `uint` or `uvecN`, as GLSL does. It used to leave the signed operand as it was, which naga rejected (#4492).
- WGSL output writes a negative integer literal stored to a `uvecN`, in a declaration, an assignment or a compound assignment, as `vecN<u32>(u32(-1i))`, as it already did for a scalar `uint`. It used to write `u32(-1)`, which naga rejected (#4504).
- WGSL output replicates a scalar argument of `min`, `max`, `step` and `smoothstep` to the vector type of the other arguments, as GLSL's `min(genType, float)`, `step(float, genType)` and similar overloads mean, including a scalar returned by `length`, `distance`, `dot` or a component-wise built-in such as `abs` or `sqrt`. It used to pass the scalar as it was, which naga rejected (#4498).
- CUDA and HIP output write `reflect`, `refract` and `faceforward` as generated helpers that follow GLSL 4.60 section 8.5, such as `cgl_float3_reflect` for `vec3` and `cgl_reflect_float` for `float`, for float, double and half scalars and float, double and `f16vec2` vectors, including calls nested in each other. Operands no helper covers, such as integer vectors or mixed vector sizes, are refused with a typed CUDA or HIP lowering diagnostic. They used to copy the calls into the output, which nvcc and hipcc rejected as undeclared identifiers, because neither toolkit declares these functions (#4473).
- An empty brace list whose destination is a scalar is that scalar's zero outside a declaration as well: assigned (`x = {};`), returned (`return {};` from a function returning `float`), or a member or element of a brace list (`S s = {{}, 1.0};`, `float a[2] = {{}, 1.0};`), as `float one = {};` already was. HLSL output used to copy the braces, which DXC rejects (`expected expression`), or refuse the program with `DirectXAggregateInitializerError`; WGSL output wrote `array()` (naga: `type can't be inferred`); and GLSL output refused it with `OpenGLAggregateInitializerError` (#3751).
- WGSL output writes HLSL's `f32tof16` and `f16tof32` with `pack2x16float` and `unpack2x16float`: `f32tof16(x)` becomes `pack2x16float(vec2<f32>(x, 0.0))`, which keeps the half in the low 16 bits, and `f16tof32(h)` becomes `unpack2x16float(h).x`, with helpers for vectors. It used to copy both calls into the output, which naga rejects (`no definition in scope for identifier: f32tof16`) (#4407).
- SPIR-V output counts the `length()` of a buffer view with an offset from the offset on, as GLSL output does: `data + 2` passed to a `buffer float*` or `StructuredBuffer<float>` parameter, or held in a local or a chain of locals, gives the buffer's `OpArrayLength` less the offset, as a signed `int`. It used to apply `OpArrayLength` to the element pointer, which spirv-val rejects (`The Structure's type in OpArrayLength must be a pointer to an OpTypeStruct`). The `length()` of a reinterpreted view, or of an array member of a buffer element, is refused with `BufferLengthUnsupportedError` (`project.translate.spirv-buffer-length-unsupported`) (#3757).
- HIP output writes CrossGL's `clockRealtimeEXT()` as `wall_clock64()`, HIP's constant-rate device clock. It used to copy the call into the output, where hipcc rejected the undeclared identifier (#4414).
- CUDA, HIP and GLSL output write the other shader clock built-ins as well: `clockARB()`, `clock2x32ARB()` and `clockRealtime2x32EXT()`, and in CUDA output `clockRealtimeEXT()`. CUDA output reads `clock64()` for the ARB clocks and the PTX `%globaltimer` register for the realtime clocks, HIP output reads `clock64()` for the ARB clocks, and the `2x32` forms return the counter as a `uint2` (low word first) that takes part in `uvec2` arithmetic. GLSL output requests `GL_ARB_shader_clock` or `GL_EXT_shader_realtime_clock` for the clocks it uses. CUDA and HIP output used to copy these calls unchanged, which nvcc and hipcc rejected as undeclared identifiers, and glslang rejected the GLSL with `required extension not requested` (#4427).
- SPIR-V output lowers the four shader clock built-ins to `OpReadClockKHR` with the `ShaderClockKHR` capability and `SPV_KHR_shader_clock`, at `Subgroup` scope for the ARB clocks and `Device` scope for the realtime clocks, as glslang does; the `2x32` forms give a `uvec2`. It used to raise `SPIRVBuiltinLoweringError` (`Cannot lower call to 'clockRealtimeEXT' to SPIR-V`). Slang, Rust and Mojo output have no counter to read and used to copy the calls unchanged; they now refuse them with `SlangLoweringRefused`, `RustLoweringRefused` and `MojoLoweringRefused` (#4427).
- A shader-scope constant or constant-buffer member passed as the first argument of a generic overload call, as in `sel(K, x)` or `sel(offset, offset)`, is a value argument. It had been read as the overload's type argument, so Metal, HLSL, Slang, CUDA and HIP output declared `K sel_K_overload0(K a)`, which their compilers reject, and GLSL, SPIR-V and WGSL output refused the program. A leading name now counts as a type argument only when it is not a local or parameter and names a built-in type, a declared structure, enumeration or type alias, or a generic parameter in scope; `cbuffer` and `buffer` block names are values (#3761).
- HLSL output declares a 1D shadow texture, which HLSL input imports from a `Texture1D` or `Texture1DArray` used only with `SampleCmp`, `SampleCmpLevel`, `SampleCmpLevelZero` or `SampleCmpGrad`, as `Texture1D` or `Texture1DArray` again. The round trip used to fail with `ValueError: DirectX texture operation 'textureCompare' requires a declared texture or image resource argument` (#2639, #2640, #2641).
- GLSL output declares a 1D shadow texture as `sampler1DShadow` or `sampler1DArrayShadow` and writes `texture`, `textureLod`, `textureGrad` and their `Offset` forms with the reference in the coordinate, as GLSL 4.60 section 8.9.2 defines them: `vec3(u, 0.0, Dref)` for `sampler1DShadow` (the second component is unused) and `vec3(u, layer, Dref)` for `sampler1DArrayShadow`. The same program used to fail with `ValueError: OpenGL texture operation 'textureCompare' requires a declared texture or image resource argument`. Metal has no one-dimensional depth texture, so it now refuses such a texture with `UnsupportedMetalFeatureError` naming the type instead of the same untyped `ValueError` (#4428).
- GLSL output counts the `length()` of a reinterpreted buffer view, such as `(uint16_t*)words` or `(float*)(words + 2)` held in a local, written inline or passed to a helper, in the view's elements: `words.length() * 2` for a `uint16_t` view of `uint` words, less the view's byte and element offsets. It used to call `.length()` on one element read from the buffer, which glslang rejects (`'length' : does not operate on this type`). A view whose element layout is unknown is refused with `OpenGLStoragePointerError` (#3758).
- HLSL and WGSL output lower a 2D shadow texture sampled through an explicit `comparison_sampler` with `texture`, `textureLod` or `textureOffset` to the depth comparison calls: `SampleCmp`, `SampleCmpLevelZero` or `SampleCmpLevel` in HLSL, and `textureSampleCompare` or `textureSampleCompareLevel` in WGSL. Outside the WGSL fragment stage, and for a literal zero LOD, WGSL uses `textureSampleCompareLevel`, since `textureSampleCompare` needs fragment derivatives. WGSL output used to refuse these calls with `WGSLComparisonSamplerUnsupportedError`, and it still refuses cube and array shadow textures and a LOD other than literal 0 with that error (#4416).
- GLSL output writes `length()` on a runtime array of structures, such as `lights.length()`, as the array's method (GLSL 4.60 section 5.9). Since 3.7.0 it had been refused with `member 'length' is not declared by structure 'Light'` (#4468).
- Slang input imports `SV_PointSize`, in any letter case, as `gl_PointSize`. It used to keep it as an unknown annotation, which translation to other targets refused (#4484).
- WGSL output writes `inverse` with a generated Gauss-Jordan helper for each matrix size and `log10(x)` as `log2(x)` times `log10(2)`, since WGSL has neither built-in. It used to write both unchanged, which naga rejects (#4471).
- SPIR-V output keeps the `LocalSize` of a compute shader that takes derivatives, and chooses `DerivativeGroupQuadsKHR` or `DerivativeGroupLinearKHR` by the shader model 6.6 rule DXC applies, refusing a size that fits neither. It used to round the size up to a multiple of 2x2, so `numthreads(4,1,1)` ran as `LocalSize 4 2 1`. Mesh outputs such as `SV_ShadingRate` take their SPIR-V built-in instead of a user location (#4471).
- Metal output writes an entry's return value to the output its semantic names: a fragment `SV_Target1` or GLSL `layout(location = 2) out` result goes to that colour attachment, and a vertex result marked `TEXCOORD0` or `COLOR1` is no longer the position. Slang input spells `SV_TargetN` as `gl_FragData[N]` and `SV_Depth` as `gl_FragDepth`, so GLSL and SPIR-V output no longer write every colour target to location 0 (#4471).
- SPIR-V input reads a built-in such as `FullyCoveredEXT` as the built-in; it used to become an ordinary input. Slang input keeps the LOD clamp of `SampleGrad` (#4471).
- WGSL output converts an `ivecN` operand of a comparison, of arithmetic or of a compound assignment to `vecN<u32>` when the other operand is unsigned, as GLSL does. It used to leave the mixed operands, which naga rejects (#4476).
- SPIR-V output refuses conflicting structure or constant definitions, a geometry stage that states no input primitive, output primitive or vertex count, and a tessellation control stage that states no output patch size, instead of filling in a default. HIP and OpenCL C input report a template argument list cut off by the end of the input as a `SyntaxError` instead of raising `AttributeError` (#4496).
- Metal output of CUDA, HIP and OpenCL C programs keeps their IEEE floating-point semantics: it starts with `#pragma METAL fp math_mode(safe)` and calls the `metal::precise` binary32 library functions, because the Metal compiler uses fast math by default and those languages do not (#4501).
- CUDA and HIP `__fadd_rn`, `__fsub_rn` and `__fmul_rn` and their `__d*_rn` forms, and code under `#pragma OPENCL FP_CONTRACT OFF` or `#pragma STDC FP_CONTRACT OFF`, are no longer fused into multiply-adds after translation. The importers mark such functions `@precise`; HLSL output declares their locals `precise`, SPIR-V output decorates their arithmetic `NoContraction`, CUDA output writes their multiplies as `__fmul_rn`, `__dmul_rn` or `__hmul_rn`, HIP output starts them with `#pragma clang fp contract(off)`, and Slang output makes their value locals precise. They used to become plain `+`, `-` and `*`, which every target was free to fuse (#4515, #4530).
- SPIR-V output decorates `NoContraction` on every operation whose value reaches a `precise` variable through assignments and initializers, as glslang and DXC do; it used to decorate only the operations written in the declaration. Metal input reads `#pragma METAL fp contract(off)`, `#pragma clang fp contract(off)` or `reassociate(off)` and `#pragma STDC FP_CONTRACT OFF` as `@precise` scopes instead of dropping them, and refuses directives CrossGL cannot express, such as `exceptions` and `eval_method` (#4529).
- A `@ workgroup_size` or `@ numthreads` attribute with one or two dimensions, such as `@ workgroup_size(64)`, sets that workgroup size on every target, with the omitted dimensions 1 as in WGSL. It used to give a 1x1x1 workgroup. One with no dimension or more than three is refused. Slang output binds the buffers of a top-level `@ compute fn` entry, the form CUDA, HIP, OpenCL C and Metal kernels take in CrossGL, as module globals; they used to be plain parameters of an ordinary function, so nothing was bound (#4516).
- CUDA, HIP and OpenCL C `round` and `roundf` round halfway cases away from zero on every target, through a helper where the target's `round` does not, and `rint`, `nearbyint` and the `__float2int_rn` family round them to even. Both used to become CrossGL `round`, which HLSL and WGSL round to even and Metal away from zero. The helper refuses binary64 operands (#4511).
- CUDA and HIP `sinf`, `cosf`, `sin` and `cos` and OpenCL C `sin` and `cos` keep their 2 ulp bound over the whole binary32 range on every target. Targets other than CUDA and HIP call generated helpers that reduce the argument by Payne-Hanek in 32-bit integer arithmetic, which fast math cannot change. Vulkan and GLSL bound only the absolute error of their own `sin` and `cos`, and fast math loosens Metal's and HLSL's (#4539).
- Metal output writes the attribute index of a member of a struct read through a device pointer as `[[attribute(n)]]`, and `[[id(n)]]` on a value-array member of an argument buffer after the member name, where the Metal compiler accepts it. The first used to be written as `[[location]]` and refused, and the second after the array extent, which the Metal compiler rejects (#4566).
- HIP input lowers HIP's own `__shfl`, `__shfl_up`, `__shfl_down` and `__shfl_xor`, which hipcc declares, as it lowers the `_sync` forms; it used to refuse them. `warpSize` becomes `int(gl_SubgroupSize)`, and a shuffle with a constant width from 1 to 64 reads lanes within its width segment (#4568, #4537).
- SPIR-V output computes `bfloat16` arithmetic in `float` and rounds the result back to `bfloat16` after each operation, so it matches the source's rounding. CUDA and HIP output combine only the active lanes (`__activemask()`, `__ballot`) in wave operations instead of assuming a full warp (#4537).
- CUDA input keeps the 32-lane warp of `__ballot_sync`, `__any_sync`, `__all_sync`, `__reduce_*_sync`, matches and shuffles: they are emulated per warp through workgroup memory, including a partial last warp. They used to become subgroup operations whose size the target chooses, such as 4 or 8 lanes on lavapipe. `__syncthreads()`, `thread_block::sync()` and `__syncwarp()` fence all memory, so storage-buffer exchanges through a barrier are ordered; they used to order workgroup memory only. `thread_block_tile<N>.sync()` becomes `__syncwarp()` for a literal power of two up to 32, and other tile syncs are refused (#4595).
- GLSL, HLSL, Metal and Slang output specialize each function that takes or declares a CrossGL generic pointer `ptr<T>`, which CUDA and HIP input write for every device pointer, to the address space it points into, with one copy per combination of spaces. They used to write `ptr<float>` unchanged, which no target compiler accepts. A pointer whose space is unknown or ambiguous is refused, and HLSL output refuses a pointer return type (#4582).
- Rust output declares CrossGL `char` as `i8` and Mojo output as `Int8`, and Mojo output promotes `char` and `short` operands to `Int32` as C does. They used to write Rust `char` and Mojo `String`, which did not compile (#4592).
- CUDA and HIP input import `size_t` and `ptrdiff_t` as 64-bit integers and `__nv_bfloat16` as `bfloat16`. They used to be 32-bit, so index arithmetic wrapped at 2^32, and binary16. Metal output passes a scalar of any integer or float width as `constant T& [[buffer(n)]]`; 8-, 16- and 64-bit scalars used to be bare kernel parameters, which the Metal compiler rejects (#4593).
- DirectX and OpenGL output lower a record whose pointer member points at a workgroup array of `f16vec2`, `f16vec4`, `bfloat16vec2` or `bfloat16vec4`, which they used to refuse. DirectX output still refuses the `bfloat16` vectors with its `bfloat16` storage diagnostic (#4626).
- DirectX and OpenGL output accept an early return on a field of a constant parameter read through `->`, as in `if (params->n <= int(tid.x)) return;`, before a subgroup barrier in a Metal kernel. The uniform control flow proof used to accept only `params[0].n` (#4661).
- Metal input parses an unnamed struct or union member together with its declarators, as in `union { ... } u;`; it used to skip it, so the program lost `u`. An anonymous member aggregate, whose members join the enclosing struct, is refused with `project.translate.metal-anonymous-member-unsupported`. Metal output passes reference arguments unconverted and declares narrow `device` and `constant` references with their own type: a `constant bfloat&` parameter used to receive a temporary the Metal compiler rejects, and a `uchar` or `ushort` reference was declared `uint&` (#4724).
- Project translation reports Metal template materialization diagnostics at the line of the original source and the start of the declaration. They used to give a position in the preprocessed text (#4681).
- GLSL output stores and loads a NaN in a half-precision lane of a union or a word view with its sign and the high bits of its payload. It used `packHalf2x16` and `unpackHalf2x16`, which need not keep a NaN's payload; Mesa stores 0x7fff (#4470).
- Project translation of Metal sources lowers a function template that only a struct's member function calls together with that struct. The template used to be instantiated after the struct's members were lowered, so Metal output called members the struct no longer declared, and HLSL and GLSL output refused them (#4470).

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
