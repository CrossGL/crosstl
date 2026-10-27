# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them.

## [Unreleased]

### Breaking Changes

- CrossGL refuses non-square `floatNxM` and `doubleNxM` matrix spellings, such as `float2x3`, with `AmbiguousMatrixSpellingError` (`project.translate.ambiguous-matrix-spelling`): MSL reads `float2x3` as 2 columns of 3 rows and HLSL as 2 rows of 3 columns, and the Metal, GLSL, HLSL and SPIR-V generators each took a different one. Write `matCxR` or `dmatCxR`. The SPIR-V importer now writes non-square matrices as `matCxR`. Square `float3x3(...)` and `float3x3(s)` now fill columns and build a diagonal matrix in HLSL output, as `mat3` does; HLSL output filled rows and repeated `s`.
- Metal output refuses `dFdx`, `dFdy`, `fwidth` and their `ddx`/`ddy`, fine and coarse forms in a compute, ray generation, mesh, object, task or amplification stage, called directly or through a helper, with `UnsupportedMetalFeatureError` (`missing_capabilities` `metal.kernel-derivatives`, `metal.mesh-derivatives` or `metal.object-derivatives`). Metal has derivatives only in fragment functions; they used to be written into the `kernel`, `[[mesh]]` or `[[object]]` function, which the Metal compiler rejects.
- Mojo output refuses a vector where a scalar is expected, such as `return v;` from a function returning `float` or `if (a == b)` on two vectors, with `MojoScalarConversionError`. It used to write the vector unchanged, which Mojo either rejects or, as an `if`, `while` or `?:` condition, reads as true when any lane is true. Use `all`, `any` or a component access to reduce the vector first.
- A type name that is neither built in nor declared by the program, as a structure, enumeration, alias, constant block or generic parameter, is refused with a typed error. Every generator used to copy it into the output.
- WGSL output refuses 64-bit integers, doubles and bfloat16, the vectors and matrices built on them and conversions such as `double(x)`, with `WGSLNumericTypeUnsupportedError`. It used to narrow `double` to `f32` and `long` to `i32`, which changed host buffer layouts, or copy the type unchanged.
- `buffer T name[]` and `buffer T name[N]` declare one storage buffer of `T` elements for every `T`, structures included, as Metal's `device T*` does. GLSL buffer block arrays (`buffer B { ... } name[N];`) keep their meaning as arrays of descriptors.
- CrossGL `sizeof` of a scalar, a 2- or 4-component vector or a sized array is folded to a constant on Metal, HLSL and GLSL. Other operands are refused with `SizeofUnsupportedError`; `sizeof` used to be written out unchanged, so GLSL rejected it and HLSL and Metal measured with their own layout rules.
- The Metal importer refuses `sinpi`, `cospi`, `tanpi`, `asinpi`, `acospi`, `atanpi` and `atan2pi`, which are exact at integers and half-integers where `sin(x * pi)` is not, and calls to functions that the source declares but never defines. Both used to reach every target as calls to undeclared functions.
- Metal indirect command buffers (`command_buffer`, `render_command`, `compute_command`) and rasterization rate maps are refused with `MetalRuntimeTypeUnsupportedError` for every target other than Metal when a stage uses them. HLSL and GLSL output used to copy the type names.
- HLSL output refuses a constant buffer larger than Direct3D's 64 KiB limit, such as a large Metal `constant T&` argument or GLSL uniform block, with `ConstantBufferTooLargeError`, and matrix varyings whose semantic indices overlap with `InterfaceLocationConflictError`. DXC used to reject the generated shader.
- The GLSL importer refuses `tensorARM` uniforms with `GLSLTypeUnsupportedError`. It used to drop them, so every target referenced an undeclared name. GLSL output refuses string literals outside `#include`, `#line` and `debugPrintfEXT` formats.
- The Metal importer refuses ray-tracing functions, and the uses of mesh and object functions it cannot import, such as `set_indices`, with `MetalImportRefused`. They used to be printed as plain helpers that still called Metal methods.
- HLSL output for the `dx10` and `dx11` profiles, as selected by `backend="dx11"`, `"d3d11"` or `"directx-11"`, refuses the `WaveSize` attribute, from HLSL `[WaveSize(...)]` or CrossGL `@WaveSize(...)`, with `DirectXWaveSizeProfileError`, because `WaveSize` needs Direct3D 12 and Shader Model 6.6 (6.8 for the range forms). It used to be written unchanged, as `[WaveSize(32)]`, which DXC rejects below Shader Model 6.6. The default `dx12` profile is unchanged.
- HLSL output for the `dx10` and `dx11` profiles, as selected by `backend="dx11"`, `"d3d11"` or `"directx-11"`, refuses mesh and amplification stages (CrossGL `mesh`, `task`, `amplification` and `object`) with `DirectXMeshStageProfileError`, because mesh shading needs Direct3D 12 and Shader Model 6.5. They used to be written unchanged, as `[shader("mesh")]` and `[shader("amplification")]` entry points that Direct3D 10 and 11 cannot run. The default `dx12` profile is unchanged.
- HLSL output for the `dx10` and `dx11` profiles, as selected by `backend="dx11"`, `"d3d11"` or `"directx-11"`, refuses ray generation, intersection, any-hit, closest-hit, miss and callable stages with `DirectXRayTracingStageProfileError`, because DirectX Raytracing needs Direct3D 12 and Shader Model 6.3. They used to be written unchanged, as `[shader("raygeneration")]` entry points that Direct3D 11 cannot run. The default `dx12` profile is unchanged.

### Added

- `crosstl doctor` lists, for each backend, the validation tools CrossTL looks for and which of them are on `PATH`. A validation that cannot run now names the missing tool.
- CrossGL `textureResidency(read, texel)` and `textureResident(code)` query sparse texture residency on GLSL, Metal, HLSL and SPIR-V.
- `metal_sizeof(x)` measures a type with Metal's layout rules. The Metal importer writes it for operands it cannot size, such as template parameters.
- Metal `[[mesh]]` and `[[object]]` functions import as CrossGL mesh and task stages.
- `gl_InstanceIndex` translates to OpenGL as `gl_InstanceID + gl_BaseInstance`, to OpenGL ES as `gl_InstanceID`, to HLSL as `SV_InstanceID` and to SPIR-V. As with `gl_VertexIndex`, HLSL's value does not include the base instance.
- Metal fixed sampler state (`constexpr sampler`) translates to OpenGL through host sampler objects. GLSL output starts with a `// crosstl-fixed-samplers:` line that lists each state with its uniforms and texture units. WebGL still refuses it.
- The SPIR-V importer reads subpass inputs as `subpassLoad`. MSL `clz` and `ctz` import through `findMSB` and `findLSB` for 8-, 16- and 32-bit operands, and Metal output lowers `findMSB` and `findLSB` instead of copying them.

### Improved

- Mojo output converts scalar boolean values to `float` and `half` through `Int`, so casts from boolean variables and comparison expressions compile.
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
- SPIR-V output lowers Metal structures with device, constant, threadgroup or thread pointer members to resource handles, as HLSL and GLSL output do. On all three targets such a member may hold a null pointer that is overwritten before every use, and a thread pointer member may point into a fixed-size local array.

### Fixed

- CrossGL accepts an empty statement, a `;` on its own, which it used to refuse with `CrossGLFunctionBodyParseError`, so every target refused a function containing one. A switch whose last `case` or `default` label is followed by no statement now leaves the switch there with `break;`; Metal and GLSL ES output used to end the switch on the bare label, which the Metal compiler and GLSL ES 3.00 refuse.
- The Slang importer writes a Slang `floatRxC` as the same CrossGL matrix, `matCxR`, as the HLSL importer does. Constructors and initializer lists fill rows, a single scalar fills every element, `m[i][j]` and the `_mRC` and `_RC` swizzles name element (i, j), `m[i]` is row `i`, and `*` of two matrices is componentwise. It used to keep the arguments and subscripts of the transposed matrix and still write `mul(a, b)` as `a * b`, so every `mul` with a matrix operand computed a different product; `row_major` and `column_major` members are now kept as `@ row_major` and `@ column_major`. A row write through a multi-component swizzle and a conversion from a value of unknown type are refused with `SlangImportRefused`.
- GLSL input names the instance it gives a buffer block without one, such as `buffer Data { float values[]; };`, so that no local, parameter or global of the program has the same name: with a local `float data` the members are read through `data_2`. The name used to be checked against global declarations only, so the local shadowed the instance and `data.values` read a member of a `float`, which does not compile.
- CUDA and HIP output apply the scalar math built-ins, such as `sin`, `exp`, `inversesqrt` and `pow`, to `f16vec3` and `f16vec4` values through per-component helpers that widen each `__half` to `float`, as in `__float2half(sinf(__half2float(value.x)))`. They used to pass the half vector straight to the float function, as in `sinf(h)`, which does not compile because no overload accepts the generated `cgl_half3` and `cgl_half4` structures.
- CUDA and HIP output expand a multi-component swizzle of a half vector, such as `e.xyz` or `t.xxyz`, through `cgl_make_half3` and `cgl_make_half4`, and a two-component one, such as `e.wy`, into a native `half2` through `__floats2half2_rn`. The swizzle is typed as a half vector, so `r + e.wzy` uses the `cgl_cgl_half3_add` helper, and an assignment to a swizzle, such as `o.xy = p` or `o.wzy += r`, writes each component. It used to be written as a member access, as in `e.xyz`, which does not compile because the generated `cgl_half3` and `cgl_half4` structures only have the members `x`, `y`, `z` and `w`.
- CUDA and HIP output expand a multi-component swizzle of a native `half2` (`f16vec2`), such as `a.yx`, into `__floats2half2_rn(a.y, a.x)`, write an assignment to one, such as `t.yx = b` or `t.xy += b`, component by component, and pass a `half2` operand of a half vector constructor as its two components, as in `cgl_make_half4(a.x, a.y, b.x, b.y)` for `f16vec4(a, b)`, through a single-evaluation helper when the operand is a call. They used to keep `a.yx` as a member access and pass `a` as one argument, which does not compile because `__half2` only has the members `x` and `y` and `cgl_make_half3` and `cgl_make_half4` take one value per component.
- GLSL input keeps an assignment or compound assignment that is an operand in parentheses, so `(b = b + 1.0) * 2.0` imports as `((b = (b + 1.0)) * 2.0)`, `-(b -= 0.5)` as `(-(b -= 0.5))` and the arms of `?:` likewise. It used to write `(b = (b + 1.0) * 2.0)`, which binds the surrounding operator to the right-hand side of the assignment, so every target computed something else or did not compile.
- CUDA and HIP output compute `%` and `mod` on half vectors per component with the same floor-based `cgl_mod_float` helper as float vectors, on the components converted to `float` and rounded back to `__half`. The helpers used to apply `%` to the `__half` components, which is not a floating-point remainder and does not compile because `__half` defines no `%` operator.
- CUDA and HIP output lower `length`, `normalize`, `distance` and `cross` on `cgl_half3` and `cgl_half4` vectors to generated helpers that compute in `float` through `__half2float`, such as `cgl_cgl_half3_length(s)`, and `distance` on `float` and `double` vectors to a `cgl_float3_distance` style helper. They used to write bare calls such as `length(s)` and `distance(a, b)`, which CUDA and HIP do not declare.
- HLSL Shader Model 6.9 `dx::MaybeReorderThread(hint, bits)` is refused with `ShaderExecutionReorderingUnsupportedError` for every target other than Direct3D, which writes the call back unchanged. The Metal, GLSL, Slang, CUDA, HIP, Rust and Mojo generators copied the HLSL spelling into their output, where each target compiler rejects it.
- CUDA and HIP reference parameters that the function writes, `T&` and `T (&v)[N]`, import as `inout`, so the writes reach the caller; a reference the function only reads and every `const T&` import by value. They all used to import by value, so writes were lost. A call that binds such a parameter to shared, global or pointed-to memory, to a local whose address is taken, or to two overlapping arguments raises `ReferenceParameterError`, as do written `T&&` and `T*&` parameters. HIP input parses `T (&v)[N]` parameters.
- CUDA and HIP output combine a vector with a scalar constructor operand such as `float(i)` or `int(x)` under `+`, `-`, `*`, `/` and `%` through the existing vector-scalar helpers, as in `cgl_float3_mul_scalar(v, float(i))`. They used to keep the native operator, as in `(v * float(i))`, which does not compile because `float3` and the other vector types define no arithmetic operators. Scalar constructor calls now have the type they construct.
- CUDA and HIP output combine a half vector with a `half` scalar under `+`, `-`, `*` and `/` through the existing vector-scalar helpers, such as `cgl_cgl_half3_mul_scalar(v, h)`. They used to keep the native operator, as in `(v * h)`, which does not compile because the generated `cgl_half3` and `cgl_half4` structures define no arithmetic operators.
- CrossGL `array<T, N>` with a nonzero extent is the array `T name[N]` in parameters, locals, struct members, globals and return types, so CUDA and HIP input, which spells C++ arrays that way, reaches GLSL, HLSL, Slang and SPIR-V as arrays. GLSL, HLSL and Slang used to write `array<i32, 2>` verbatim and SPIR-V refused it. `array<T, 0>` and `array<T>` keep the standard-array type. SPIR-V output passes a by-value array held in a struct member or an array element through its own `Function` variable, since `OpFunctionCall` takes only memory object declarations; spirv-val rejected the access chain it used to pass.
- CUDA and HIP output splat a half vector constructor whose only argument is a component of a native `half2`, such as `f16vec3(b.x)` for an `f16vec2 b`, into `cgl_make_half3(b.x, b.x, b.x)`. It used to be written as `cgl_make_half3(b.x)`, which does not compile because `cgl_make_half3` takes three components.
- WGSL output discards the value of a call statement to a `@must_use` built-in such as `min(a, 4u);` or a structure constructor such as `Pair(a, b);` with a phony assignment, `_ = min(a, 4u);`. The call used to be written as a statement, which WGSL rejects. Atomics, barriers, `textureStore` and functions the shader declares are written unchanged.
- OpenGL and WebGL output accept `vals[k++] = 1.0; vals[k] = 2.0;`. `ASTNode.child_nodes` yielded a child kept under two field names, such as `ArrayAccessNode.index` and `index_expr`, once per name, so the GLSL interval analysis applied the increment twice, took `k` to be 2 and refused the in-range `vals[k]` as a constant index out of range. Each child is now yielded once (#3233).
- WGSL output writes a one-scalar matrix constructor such as `mat2(s)` or `mat2x3(s)` column by column with `s` on the diagonal and zeros elsewhere, as in `mat2x2<f32>(vec2<f32>(s, 0.0), vec2<f32>(0.0, s))`, including a negated argument such as `mat2(-1.0)`, and converts an integer argument to the element type. It used to be copied as `mat2x2<f32>(s)`, which WGSL rejects because it has no diagonal matrix constructor. An argument with side effects, which would be evaluated once per diagonal element, is refused with a message asking for a local variable.
- CUDA and HIP output splat a vector constructor whose only argument is a scalar cast, such as `ivec3(int(x))`, through the `cgl_int3_splat` helper. It used to be written as `make_int3(int(x))`, which does not compile because `make_int3` takes one argument per component.
- The Slang importer keeps `out` and `inout` on the parameters of every function, not only of entry points, so a helper's writes to an `inout int k` reach the caller (#3215). A `constref` parameter imports as the value it reads. A `ref` parameter, which names the caller's storage, is refused with `SlangImportRefused`; it used to be read as a type named `ref` and produced invalid CrossGL.
- CUDA and HIP output keep `sinpi` and `cospi`, which both math libraries provide exactly at integers and half-integers: `sinpif` and `cospif` for `float`, `sinpi` and `cospi` for `double`, per component for vectors. They used to be refused with `MetalMathFunctionUnsupportedError`, whose message also named MSL when the call came from CUDA or HIP source.
- Mojo output of a component-wise vector comparison such as `a < b` on `vec2` operands is a `SIMD[DType.bool, 2]` mask written with `a.lt(b)`, and `any` and `all` of a boolean vector become `reduce_or()` and `reduce_and()`, ignoring the padding lane of three-component vectors. A `bvec2` initialized from a comparison used to be refused.
- OpenGL and WebGL output copy an unqualified array parameter in, as GLSL and every other target do, so a helper that writes its array parameter no longer changes the caller's array (#3201). The generator gave every written fixed-array parameter `inout`, because asking whether a helper writes its array parameter also marked it for writeback. Writeback now stays with callers that pass caller-visible storage such as `&local`, and with the helpers they forward it to.
- Mojo output splats a `bool` into a boolean vector with the `fill` keyword, as in `SIMD[DType.bool, 2](fill=s)` for `bvec2(s)` and for a `bool` compared with a `bvec2` on either side. It used to write `SIMD[DType.bool, 2](s)`, or pass `s` unchanged on the right of a comparison, which Mojo rejects because `Bool` is not `Scalar[DType.bool]`. A comparison or `&&` / `||` of scalars is now typed `bool` rather than its operand type, so `bvec2(x < 1.0)` is written `SIMD[DType.bool, 2](fill=(x < 1.0))` instead of calling `.cast[DType.bool]()` on a `Bool`, `a && (x < 1.0)` on a `bvec2 a` splats the comparison like a `bool` operand, and `mix(a, b, x < 0.5)` on scalars selects `b` or `a` instead of passing a `Bool` to `lerp`.
- Mojo output writes `!` of a boolean vector, such as `!(a == b)`, as the lane-wise `~` and types the result as a mask of the same width, so `any`, `all` and `mix` read it per lane and `bool r = !(a == b);` is refused with `MojoScalarConversionError`. It used to write `not`, which Mojo rejects where a `SIMD[DType.bool, N]` mask is expected and, assigned to a `bool`, computes `!any(a == b)`.
- Mojo output writes a float or integer vector built from a `bool`, such as `vec2(s)`, `vec2(s, x)` or `ivec2(s)`, as `SIMD[DType.float32, 2](Float32(Int(s)))`, `SIMD[DType.float32, 2](Float32(Int(s)), x)` and `SIMD[DType.int32, 2](Int32(Int(s)))`, and bool vector components the same way, as in `Float32(Int(b[0]))`. It used to write `(s).cast[DType.float32]()`, which Mojo 1.2.0 rejects because `Bool` has no `cast` method.
- Mojo output combines boolean vector masks under `&&` and `||` per component with `&` and `|`, as in `(x).lt(y) & (y).lt(z)`. It used to write `and` and `or`, which Mojo accepts but evaluates as a whole operand chosen by whether any lane of the left mask is true, so `a && b` with `a = (true, false)` and `b = (true, true)` gave `(true, true)`. A scalar `bool` operand of a mask is splatted to its lanes, as in `(a & SIMD[DType.bool, 2](fill=s))`, and two scalar `bool` operands keep `and` and `or`.
- CUDA and HIP output declare a CrossGL `out` or `inout` parameter as a C++ reference, `float& x` or `float (&v)[2]`, so the callee's writes reach the caller (#3159). They used to declare it by value and drop the writes. A call that passes one variable to an `out` or `inout` parameter and to another parameter that reaches the caller, where references and copy-out differ, raises `CudaLoweringRefused` or `HipLoweringRefused`.
- WGSL and Slang output keep the named fields of a braced structure initializer such as `Pair { b: y, a: x }`, in member order, and refuse a missing, unknown or repeated field. They used to construct the structure with no arguments, which zeroed every field.
- WGSL output gives `imageSize` the shape GLSL defines: `i32` for 1D images, `vec3<i32>` for 3D images, and `vec3<i32>` with the layer count last for 2D array images, or the number of cubes for cube array images. It used to write `vec2<i32>(textureDimensions(image))` for every image, which naga rejects for 1D, 3D and array images.
- A for-loop initializer that declares several variables, `for (int a = 0, b = n; a < b; a++, b--)`, reaches GLSL, WebGL, MSL, Slang, CUDA and HIP output as one declaration and SPIR-V output as both declarations. GLSL, WebGL, MSL and Slang output used to write the parser's list of declarations into the loop header, CUDA output wrote the list of updates, HIP output ran the updates once before the loop, and SPIR-V output refused the program. The GLSL interval analysis also kept the declared variables at their initial values inside such a loop and dropped branches it took to be unreachable. Declarations of different types raise `ForInitializerError` (#3252, #3253, #3255, #3257, #3258, #3261, #3278).
- CUDA and HIP output writes `min` and `max` of integer operands as the integer intrinsics instead of `fminf` and `fmaxf`, which converted the operands to `float` and rounded values above 2^24, and writes `double` operands as `fmin` and `fmax`. Operands of different integer types are cast to the type they promote to, as `min(static_cast<unsigned int>(i), 32u)`.
- WGSL output gives resources without an explicit binding the lowest free slots in their bind group, as GLSL and HLSL output do. Explicit bindings in any group used to push them higher: next to `cbuffer Params @binding(3)`, two storage buffers without a binding took slots 4 and 5 instead of 0 and 1.
- Slang, CUDA and HIP output write a nested `{ ... }` block, such as a block in a function body or a braced `case`, as a block. Slang output used to write `BlockNode(statements=2);`, which slangc rejects, and CUDA and HIP output dropped the braces, so declarations in the block redeclared outer names. A declaration in the block ends with it, so a shadowed outer variable keeps its own type after the block (#3259, #3270, #3272).
- Metal output of a mixed-signedness vector operation declared with an integer vector type, such as `int4 r = a + b` on an `int4 a` and a `uint4 b`, keeps both operands in the common type, as in `int4 r = int4(uint4(a) + b);`. It used to convert the right operand back to the declared type, as in `int4(uint4(a) + int4(b))`, which `metal` rejects because the operands are mixed again. `int4 q = a + g` on a `float4 g` is likewise written `int4(float4(a) + g)` instead of `int4(float4(a) + int4(g))`.
- Metal output of matrix shape conversions such as `(float3x3)m`, of `as_type<long>` and `as_type<ulong>`, and of fragment colour attachment inputs compiles, and HLSL `GetDimensions` assigns its stride output on Metal.
- CUDA, HIP and OpenCL C input keep a nested `{ ... }` block and a braced `case` body as a CrossGL block. The parsers spliced it into the enclosing block, so a declaration in it redeclared an outer name, or one in another `case`, and a use after the block read the inner variable (#3256, #3273, #3274, #3276). A HIP union shadowed in a nested block, which was refused for that reason, now translates.
- HLSL output of Metal sources gives `half` fragment returns `SV_Target`, declares `half` textures without a stray array size and drops `<TargetConditionals.h>`.
- The runtime loader manifest's `targetProfiles` for HLSL artifacts, from `directx_target_profiles_for_source`, is `["directx-12"]` when the output has ray tracing stages (`raygeneration`, `intersection`, `anyhit`, `closesthit`, `miss`, `callable`), mesh or amplification stages, or wave and quad intrinsics such as `WaveActiveSum` and `QuadReadAcrossX`. It used to also list `directx-11`, which cannot run them and whose `dx11` HLSL profile refuses them.
- GLSL, Slang, CUDA, HIP and OpenCL C input write a C-family octal literal such as `017` as the CrossGL octal `0o17`. They copied it unchanged and CrossGL read it as decimal, so `017 + 010` computed 27 instead of 23 in every target (#3282). A leading-zero integer with an 8 or a 9 raises `IntegerLiteralError`.
- An early `return;` in a GLSL `main` that writes stage outputs returns those outputs on every target.
- CUDA output writes CrossGL scalar casts such as `f64(x)` with C type names.
- Metal output converts the result of a vector expression to the declared type rather than each operand, so `float4 x = a / int4(2, 2, 2, 2)` stays an integer division and `int4 x = b / 2u` an unsigned one (#3166). A vector whose component type ranks below a scalar operand's is converted first, `float4(a) * 0.5`, because MSL would convert the scalar to the vector's component type instead; the result then has the common type, so `int4 x = a & 4u` converts the `uint4` result back. A floating literal keeps a `half` vector's precision: `h * 2.0` stays a `half4` product.
- CUDA output passes the address of the target to block- and system-scoped atomics such as `atomicAdd_block` and `atomicInc_system`, as it already did for `atomicAdd` and `atomicInc`.
- CUDA and HIP output copy an `in` array parameter that the function writes, such as an HLSL `float v[2]` parameter, into a local array at function entry, so the caller's array keeps its values as it does in HLSL. The parameter used to be emitted as a C array, which decays to a pointer, so writes in the callee changed the caller's array. A parameter counts as written when the function assigns, increments or takes the address of an element, writes through it with `*v`, `v->a` or `v + 1`, or passes the array or an element to an `out` or `inout` parameter, a non-const pointer parameter, an unqualified array parameter or `frexp`, `modf` or `sincos`, including through an explicitly specialised generic call such as `fill<int>(v, 7)`. Unwritten `in` arrays, `out` and `inout` arrays, arrays without a qualifier and arrays of pointers, references, structured buffers or other resources are unchanged; a struct the program declares is copied whatever its name, such as `Texture2DParams`.
- Metal output renames a user function named like a Metal Standard Library function, such as `fwidth`, `min` or `simd_sum`, with every overload and every call to it (`fwidth_fn`). `using namespace metal;` made a call to it ambiguous, which the Metal compiler rejects (#3156).
- A program with several stages of one kind, such as a Metal library with several vertex functions, has every stage rewritten, not only the first.
- Translating a malformed shader no longer takes time that grows much faster than its length.
- SPIR-V output evaluates a subscript index that writes, such as `k--`, `next()` or `(j = k + 1)`, once when a declaration's initializer reads an element (`float a = t[k--];`, `S s = data[k++];`, `float x = s[k++].x;`) and when an assignment is used as a value (`a = (t[k++] = x);`). The emitter first checked whether the initializer aliased a resource, or stored through the assignment target, and then lowered the access again, so the side effect ran twice and the read used the wrong element (#3294).
- The Metal check that refuses a derivative a kernel, mesh or object function reaches through a helper follows every overload a call can select: those that accept its argument count, narrowed by the argument types when they decide. It followed only the last overload of the name, so a derivative in an earlier overload reached the Metal compiler, which rejected the kernel (#3168).
- An assignment used as a value keeps its own parentheses: `int f = (e = 4) + 1;` used to become `e = 4 + 1` on several targets, which assigns 5.
- CUDA, HIP and Metal output keep the parentheses around the operand of a postfix `++` or `--` unless it is a name, member access, index or call, so `(*p)++` stays `(*p)++`. It used to be written `*p++`, which increments the pointer and discards the value because postfix `++` binds tighter than unary `*`, and does not compile for an array, as in `(*v)++` on a `float v[2]`.
- SPIR-V output short-circuits `&&`, `||` and `?:`. The right operand and both arms used to run unconditionally.
- GLSL output declares `long`, `ulong`, `size_t` and `ptrdiff_t` as 64-bit integers instead of 32-bit ones, which changed arithmetic and block layouts; GLSL ES and WebGL refuse them. SPIR-V output converts `size_t(x)` and `ptrdiff_t(x)` and widens `uint` without sign extension.
- 8- and 16-bit integer vector arithmetic wraps at its own width on GLSL and Metal, and Metal widens native `ushort2` and similar fields and buffer elements when it reads them.
- Storage buffer arrays translate to GLSL, SPIR-V, HLSL, WGSL, Slang and Metal. Sized arrays on HLSL used to be `static` arrays whose stores never reached memory, and unsized arrays of structures were refused on HLSL and WGSL.
- A GLSL entry point that writes one of its inputs, built-ins included, writes a local copy instead of the read-only input.
- HLSL entry points that write stage outputs through `out` and `inout` parameters translate to GLSL, HLSL, Metal, SPIR-V, WGSL and Slang, and `SV_Target<N>` reaches colour attachment N on GLSL, WGSL, Slang and Metal. Multi-stage shaders written this way used to fail, and GLSL and WGSL numbered attachments in declaration order.
- WGSL output of a `switch` has one `default` arm and keeps fallthrough by merging case bodies, writes `discard;`, and converts scalar operands of vector stores, shifts and bitwise operators, which naga used to reject.
- WGSL output declares the integer and unsigned 1D, 3D, cube and cube array samplers, such as `isampler3D` or `usamplerCube`, as `texture_3d<i32>`, `texture_cube<u32>` and so on. It used to declare them under their CrossGL names, which naga rejects. `sampler2DRect`, `samplerBuffer` and the other samplers with no WGSL texture type are refused with a diagnostic, as `sampler2DMS` already was.
- An HLSL vertex input with `SV_Position` is a regular vertex attribute on GLSL, Metal, WGSL and SPIR-V. Metal and WGSL rejected it, and SPIR-V read the `gl_Position` output instead.
- Function-scope `static` variables keep their value across calls on every target, and HLSL output keeps the initializers of static globals.
- WGSL output of `texture()`, `textureLod()`, `texelFetch()` and `texelFetchOffset()` on 2D array and cube array textures passes the layer as its own `array_index` argument, as WGSL requires. It used to leave the layer inside the coordinate, so naga rejected the output. Sampled lookups round the layer with `floor(layer + 0.5)`, as GLSL does.
- Vector `&&` and `||` combine per component on every target. HLSL 2021 and WGSL output used to be invalid, and GLSL refused it.
- Metal output of HLSL vertex inputs that mix attributes with built-ins such as `SV_VertexID` compiles, and an `int` `SV_VertexID` is accepted.
- Metal fragment parameters whose members carry `[[color(n)]]` read colour attachments on every target, or are refused where the target has no framebuffer fetch. They used to become ordinary inputs. A default-constructed Metal `sampler` gets MSL's default sampler state.
- The Metal importer reads `m.columns[i]` of a matrix as the column `m[i]`. The `columns` array used as a value is refused with `MetalMatrixColumnsError`.
- HLSL output stores bfloat16 vectors as per-lane `uint` payloads, writes `fwidthFine` and `fwidthCoarse` with fine and coarse derivatives, and the HLSL importer maps packed 8-bit structured buffer elements.
- The HLSL importer reads `round` as CrossGL `roundEven`, because HLSL rounds halfway cases to the nearest even integer. Metal output used to write `round` and CUDA and HIP output `roundf`, which round them away from zero, and GLSL output `round`, which leaves them to the implementation, so `round(2.5)` gave 3 instead of 2. Metal output now writes `rint`, converting an integer operand to `float` because `rint` has only `float` and `half` overloads, as in `rint(float(i))`; CUDA and HIP output write `rintf`, or `rint` for a `double` operand, and round `float` and `double` vectors per component; GLSL output writes `roundEven`; Slang output writes `rint`, which rounds halfway cases to even on every Slang target, where Slang `round` leaves them to the target; Rust output calls `round_even` instead of `round`; and Mojo output calls its `round_even` helper, which now has overloads for `Float16`, `Float32` and `Float64` scalars and 2- and 4-wide vectors, converting an integer operand to `Float32` and spelling a literal operand as `Float32(2.5)`. A function the shader defines named `round` is still called when the call's argument count fits its parameters.
- A scalar swizzle such as `s.xxx` becomes a vector constructor on Metal, WGSL and GLSL ES, which do not accept it.
- Slang output reads `gl_NumWorkGroups` as `WorkgroupCount()`, and Metal output infers the type of a shader-scope `let` from its initializer.
- Metal `device T&` and `const device T&` arguments that reference a structure read the buffer on every target. HLSL, GLSL and SPIR-V used to declare them as stage inputs, so the shader read vertex attributes or nothing.
- Metal `[[vertex_id]]`, `[[instance_id]]` and `[[primitive_id]]` arguments keep their `uint` type, so arithmetic on them wraps as in MSL. They used to import as `int`.
- OpenCL C `barrier(flags)` and `work_group_barrier(flags[, scope])` order the memory their flags name. `CLK_GLOBAL_MEM_FENCE` and `CLK_IMAGE_MEM_FENCE` put the CrossGL buffer or image fence on each side of the execution barrier, as in `threadgroup_barrier(mem_flags::mem_device)` for Metal, `DeviceMemoryBarrierWithGroupSync()` for HLSL and `storageBarrier()` around `workgroupBarrier()` for WGSL. They used to lower to `workgroupBarrier()` whatever the flags said, which orders work-group memory only, so a write to a global buffer before the barrier could be read stale after it. Flags that are not only `CLK_` names, such as a variable, order all memory. A `memory_scope` wider than the device is refused with `OpenCLFenceScopeUnsupportedError`.
- A GLSL local whose initializer reads an outer variable of the same name, as in `float scale = scale * 3.0;`, is renamed on HLSL, Metal, CUDA, HIP and Slang, where the initializer would read the new, uninitialized variable. The GLSL importer no longer reads a local that shadows a member of an anonymous buffer block, or of a uniform block it flattens, as that member.
- Row-major matrix struct members (`@ row_major`, from GLSL `layout(row_major)` block members, HLSL and Slang `row_major` and SPIR-V `RowMajor`) keep their bytes on GLSL, WebGL, SPIR-V, Metal, WGSL, CUDA, HIP, Rust and Mojo. Those targets declare the member as its column-major transpose and rewrite every read and write; HLSL and Slang do so for GLSL buffer blocks only. The member used to be declared column-major, so reads, writes and the offsets of later members changed. Uses the transposed storage cannot keep, such as writing one column, raise `RowMajorMemberError`.
- GLSL output zero-initializes arrays with one constructor argument per element, and refuses an array whose size it cannot evaluate with `OpenGLZeroValueError`. HLSL output writes a one-argument vector constructor of unknown width as a cast, which splats a scalar.
- Locals named after built-ins that GLSL and HLSL output call, such as `texture`, `textureSize`, `mod` and `imageSize`, are renamed so they no longer hide those functions.
- HLSL output adds no second `SV_Position` parameter to a fragment entry whose input structure has one, declares `int` members that carry `SV_RenderTargetArrayIndex`, `SV_ViewportArrayIndex`, `SV_PrimitiveID` or an index system value as `uint`, and gives members named `view`, `layer` or `viewport` no system-value semantic. A Metal varying `float3 view` used to carry the view index.
- Early fragment tests reach HLSL as `[earlydepthstencil]`, SPIR-V and Metal output, and the HLSL importer reads `[earlydepthstencil]`. Only GLSL output kept them.
- The HLSL importer keeps a value-returning function marked `[numthreads]` as a helper, as DXC does, and GLSL output converts a scalar returned by a `float4` pixel entry to the output's type.
- The Slang importer keeps an entry point's `[WaveSize(...)]` attribute, in the single and the range forms, as CrossGL `@ WaveSize(...)`, as the HLSL importer does. It used to drop it without a diagnostic, so DirectX output ran with whatever wave size the driver picked. Slang and SPIR-V output now refuse the attribute, as they already did for HLSL input.
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
