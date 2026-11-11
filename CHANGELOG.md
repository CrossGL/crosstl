# Changelog

Notable changes to CrossTL for people who install and use it, newest first. Internal refactors, test and CI changes are left out; the commit history has them.

## [Unreleased]

### Breaking Changes

- GLSL, HLSL and Metal output refuse a built-in type of another language that the target cannot spell, such as Mojo `Float16`, CUDA `__half` or `dim3`, HLSL `HitObject` or an integer matrix in GLSL and Metal, with the target's `LoweringRefused` ("the built-in type 'X' has no <target> spelling"). Such a name used to be written unchanged, and the target compiler rejected the output. HLSL output also refuses GLSL `atomic_uint` and HLSL and GLSL output refuse Metal `visible_function_table`, and Metal output refuses a Mojo `UnsafePointer[T]` parameter, which were all copied into the output. The refusal is decided on the rendered program: a name the generator maps but a later lowering replaces, such as GLSL's `texture2DShadow` for a texture used with a comparison sampler, is not refused, and neither is a type the program or the output declares under that name. GLSL and Metal output now spell HLSL's sized `float32_t`, `float64_t` and `float16_t` scalars, vectors and matrices as `float`, `double` and `half` types (`float32_t4` is `vec4` in GLSL), and GLSL output spells sized integer vectors such as `int16_t2` as the matching `ivec`/`u64vec` type.
- Every target refuses a program whose function body reads or writes a plain name it never declares, with `UndeclaredIdentifierError` (`project.translate.undeclared-identifier`), which names the identifier and the function. A name declared in a block is out of scope after the block, a loop variable after its loop, and a local before its declaration. HLSL, GLSL and Metal input kept such a name, as `r` in `dst[0] = r;`, and every generator copied it into the output, where it failed to compile or named a different variable with the same spelling. Functions no entry point reaches are not checked, because the generators drop them.
- Every target refuses a program whose live code calls a plain function name that is neither declared by the program nor a CrossGL built-in or a built-in of a language CrossTL imports, with `UndeclaredFunctionError` (`project.translate.undeclared-function`), which names the function. Metal, HLSL and GLSL output copied such a call, as `undeclaredHelper(x)`, into the target, where it failed to compile or called a different function with the same spelling; only SPIR-V output refused it. Qualified and member callees, and functions no entry point reaches, are not checked; nor, for Metal, HLSL and GLSL output, are generic structures that live code never names, because those generators emit only the instantiations a program names.
- Metal and Slang output refuse a constant-buffer member placed by `@offset` or `@align` where their declaration would put it elsewhere, with `MemberPlacementError` (`project.translate.member-placement-unsupported`); they used to drop the placement (Metal) or write it as a semantic (Slang). HLSL and GLSL output refuse such placements in the struct of a uniform `ConstantBuffer<T>` that declares no layout, and GLSL output refuses a push-constant offset its `layout(offset)` cannot write. HLSL input refuses a `[[vk::push_constant]]` struct whose layout under DXC's default rule CrossGL cannot record (a nested struct DXC lays out differently from std430, a struct that needs offsets and is laid out elsewhere too, or 16-bit members) with `HLSLPushConstantLayoutError`.
- A helper's `buffer T name[]` storage-buffer parameter takes the structured-buffer form of the buffer its caller passes: HLSL and Slang output pass the resource, GLSL and SPIR-V output specialize the helper for each buffer it receives, and Metal output passes a device pointer. HLSL, GLSL, SPIR-V and Slang output had printed an unsized array parameter, which DXC, glslang, spirv-val and slangc reject. Such a parameter with an std140 layout, or with elements other than 32-bit scalars, two- or four-component vectors of one, or structs of them without std430 padding, is now refused with `StorageBufferParameterUnsupportedError` for every target, since those elements have no single stride across std430, HLSL structured buffers and MSL.

### Fixed

- Importers reject malformed input instead of hanging or failing with an internal error. A construct that the end of the file cuts off (`layout(` in CrossGL, a CUDA template argument list, a Slang parameter list) raises `SyntaxError` where every parser kept consuming the end of input. A `#` after other tokens on an HLSL line, and the Mojo transfer operator on a literal (`5^`, which `mojo build` rejects), raise `SyntaxError`. Slang input refuses a floating-point literal outside its type's range (`1e400`, `1e-400`, `65520.0h`), which slangc converts to `inf` or `0`, with `project.translate.slang-float-literal-unrepresentable`; a literal such as `1e4294967296` exhausted memory. SPIR-V input refuses a module that `spirv-as` or `spirv-val` rejects with `project.translate.spirv-invalid-module`: an `<id>` operand that no instruction defines, a branch or merge instruction with missing or extra operands, a label operand that is not a block of the function, a result `<id>` that two instructions define, or a type that names a type or constant declared after it (other than through `OpTypeForwardPointer`).
- SPIR-V output passes `out` and `inout` arguments by value-result, as GLSL and HLSL define them: each such parameter has its own variable, an `inout` argument is copied in when it is evaluated, and the value is copied back through the l-value fixed then, after the call. The parameter used to be bound to the caller's variable, so `scale(m, bump(m))` read `m` after `bump(m)` had changed it, and an `inout` parameter the callee never writes did not undo a later argument's write. Metal `thread`, `device` and `threadgroup` reference parameters keep reference semantics.
- SPIR-V output places the members of a struct a constant buffer holds, at any depth, at the bytes their `@offset` or `@align` declare: `struct Inner { float s; vec4 v @offset(32); }` held by a `cbuffer` or a `ConstantBuffer<T>` decorates `v` with `Offset 32`, where it wrote `Offset 16` and was refused. An array of `ConstantBuffer<T>` lays out each element block as one `ConstantBuffer<T>` does, with the placements of `T` and a declared `@buffer_layout(hlsl_cbuffer)` or `metal`; it used to drop them. HLSL and GLSL output refuse an array of constant buffers whose declared layout or placements their own constant-buffer rules would move, with `ConstantBufferArrayLayoutError` (`project.translate.constant-buffer-array-layout-unsupported`).
- GLSL, GLSL ES and HLSL output place the members of a struct a constant buffer holds, at any depth, at the bytes their `@offset` or `@align` declare, as HLSL `[[vk::offset(N)]]` on a struct member does: `struct UBO { [[vk::offset(272)]] float alpha; }; cbuffer ubo { UBO ubo; }` reads `alpha` from byte 272. GLSL `layout(offset)` and HLSL `packoffset` place only the members of a block, so such a buffer is declared as 16-byte registers and each member is copied out of its bytes when the stage starts; all three outputs used to refuse it with `MemberPlacementError`. A member without `@offset` in such a struct follows the member above it by the buffer's rules on every target. A push-constant block that holds such a struct is still refused on GLSL and HLSL output.
- HLSL output computes `fmod`, `frexp`, `modf` and a floating-point `%` with integer helpers, so each result is exact and carries the sign of `x`, as in C, GLSL and MSL. DXC lowers `fmod` through a fast division, so `fmod(-5.5, 2.0)` read `-1.4999998` on WARP; `frexp` returned a positive significand for a negative value and zero for a subnormal one; `modf` returned a positive zero for a negative integer and NaN for an infinity; and a double `%` failed to compile. Metal output of `%` between built-in calls on floats, such as `abs(x) % abs(y)`, writes `fmod`, where it had written `%`, which the Metal compiler rejects for floating-point operands.
- GLSL and GLSL ES output place the members of a push-constant struct that carry `@offset` or `@align`, such as HLSL `[[vk::offset(12)]] float roughness;` in a `[[vk::push_constant]]` struct. The Vulkan block writes every member's `layout(offset = N)`, and the uniform block that stands for it outside Vulkan pads the members with floats and `vec4` arrays so that std140 puts them at the same bytes. Such a block was refused with `PushConstantBlockError`; a member that std140 cannot lay out at its push-constant offset, or one at a relaxed offset GLSL cannot write, still is.
- Metal output lays a push-constant block, `ConstantBuffer<T>` or `uniform T` out by its push-constant layout (std430 unless it declares std140 or scalar), as the host pushes it. It used the natural MSL layout, so `vec3 v; float s;` read `s` at byte 16 instead of 12. Metal output also places the members of a struct that a planned constant block holds by their `@offset` and `@align`: HLSL `[[vk::offset(272)]] float alpha;` in a struct inside a `cbuffer` was read at byte 0. Push-constant blocks whose members carry offsets, such as DXC's `[[vk::offset(N)]]` and relaxed offsets, are no longer refused.
- HLSL input subscripts an array member of a structured-buffer element as an array. With `RWStructuredBuffer<E> e` and `float3 v[2]` in `E`, `e[1].v[i] = float3(4.0, 5.0, 6.0);` stored only the first component (`(vec3(4.0, 5.0, 6.0)).x`) on every target, and `e[0].m[1] = r[1].m[0];` for `float2x2 m[2]` was written as a row of one matrix. The member's array dimensions are now looked up in the buffer's element struct.
- Metal output reads and writes matrices held by HLSL structured buffers, as the element type (`StructuredBuffer<float3x3>`) or as a struct member. A structured buffer packs a matrix's columns (its rows, for `row_major`) one after another at four-byte alignment, so `float3x3` takes 36 bytes; Metal output holds such a matrix as its words and reads and writes it through helpers, where it used to refuse the buffer. A store to part of such a matrix is refused. A subscript of a structured buffer's vector element, `pos[1][2]`, reads one component, where Metal output had converted the element twice, and `cnt[1][2] = v` stores one.
- OpenGL output holds an HLSL structured buffer as 32-bit words when its struct element has an array member, or a member struct with one, and the HLSL and std430 layouts differ, as SPIR-V output does: `struct { float s; float3 b[2]; uint k; }` keeps `b` at byte 4 with a 12-byte stride and `k` at byte 28. Such buffers had been declared unchanged under std430, which puts `b` at byte 16 with a 16-byte stride, so the shader read and wrote other bytes than the host. An element whose layout differs because of a matrix member, such as `struct { float s; float3x3 m; }`, is refused with `StructuredLayoutUnsupportedError` for OpenGL as it already was for SPIR-V.
- HLSL input reads `StructuredBuffer<float2x3>` as a buffer of `mat3x2`, where the element type had kept the HLSL spelling, which every output refused as ambiguous. `StructuredBuffer<row_major float2x3>` is refused, because DXC's DXIL output ignores the qualifier and its SPIR-V output applies it. With row-major packing (`matrix_packing="row_major"`, DXC `-Zpr`), an explicit `row_major` or `column_major` qualifier now gives the CrossGL orientation of the transposed matrix the importer records; Metal output had read such constant-buffer members with their rows and columns swapped.
- Metal input materializes a function template for each call that deduces its arguments when its parameters or body name a member type that an explicit specialization redefines, such as `opmath_t<T>` for `template <typename T> using opmath_t = typename OpMathType<T>::type;` with `template <> struct OpMathType<half> { using type = float; };`. The generic function had used the primary template's member type for every argument, so `scale(x, factor, w)` on `half` values computed in `half` and returned infinity where the source computes `60000 * 2` in `float`. Alias templates and structs that a using-declaration such as `using lib::math::opmath_t;` names are found by unqualified type lookup; such an alias had been left unresolved, and a body-local alias of it was dropped. An unresolved body-local alias in a generic function is refused with `project.translate.metal-local-type-alias-unresolved`.
- Metal input keeps every overload of a class template member that a sibling member calls through the implicit object. With `post_in(complex_T)` and `post_in(T)` in `template <typename T> struct RW`, `load` calling `post_in(buf[0])` left both `RW_float__post_in` helpers out as unused, so Metal output named an undeclared `post_in` and failed to compile. A function template whose body names a type through an expression of its parameters, such as `using phase_T = decltype(M_PI_F * T(0));`, is no longer refused when only members of class template specializations call it: each of them instantiates it, and its generic definition, which CrossGL cannot spell and nothing calls, is left out. A generic definition the output still names stays refused with `project.translate.metal-local-type-alias-unresolved`.
- SPIR-V output stores an array value into a Function variable before it passes it to a by-value array parameter. A program-scope `const float W[4]`, a row of a `const` array of arrays, a member of a `const` struct, an array constructor and an array returned by a call had been passed as the value itself, and `spirv-val` rejected the `OpFunctionCall` because its parameter is a Function pointer.
- HLSL output evaluates call arguments from left to right, as GLSL, MSL and SPIR-V do. DXC evaluates the arguments of function and intrinsic calls from right to left, so `two(x, bump(x))` passed the updated `x` first and `max(bump(x), x)` read `x` before `bump` ran. A floating-point `%`, which HLSL output computes with an exact `fmod` helper call, orders its operands the same way. When one argument's side effect can change what another reads, the arguments up to that side effect are evaluated into temporaries, in order, inside the expression where the call stands, so a call under a condition or a short-circuit operand still runs only there. An `inout` argument is copied in at its turn and back after the call, an `out` argument's location and an atomic's object and expected value are fixed at their turn, and a call that cannot be ordered this way is refused with `project.translate.directx-lowering-unsupported`.
- Slang output places `@location(N)` stage varyings. A fragment colour output takes `SV_Target<N>` and any other varying `TEXCOORD<N>`, where the bare `: location` had lost `N` and was no valid pixel shader output for DXC. Varyings also carry `[[vk::location(N)]]`, since slangc numbers SPIR-V locations in declaration order: GLSL inputs and outputs with gaps or out of order, and an HLSL `SV_Target2` after `SV_Target0`, reached other locations.
- Slang output gathers a vertex or fragment stage's `in` and `out` variables, and module-scope `out` declarations, into the entry point's input and output structures, as Metal and HLSL output do: `in vec2 uv @ TEXCOORD0;` becomes `float2 uv : TEXCOORD0;` in `FragmentInput`, read as `input.uv`, and `out vec4 color @ gl_FragColor;` becomes `float4 color : SV_Target;` in the returned `FragmentOutput`. An entry point that also declares its own interface parameters is refused with `StageInterfaceUnsupportedError`, as it is for Metal and HLSL. Slang output used to declare these variables as globals, which Slang reads as uniform shader parameters: an input read a constant buffer instead of the interpolated value (slangc warns `E39019`), and a write to an output failed with `E30011: left of '=' is not an l-value` (#3655).
- GLSL input applies a shader storage block's `row_major` (on the block or a `layout(row_major) buffer;` default) to its matrix and struct members, as it already did for uniform blocks (GLSL 4.60 section 4.4.5). Every output had read such a block's matrices column by column, so `m[0]` returned a row.
- Uniform blocks and constant buffers may hold multidimensional arrays. A GLSL std140 block with `float a[2][3]`, `S s[2][2]` or `mat3 m[2][2]` reaches HLSL output (with `packoffset`s) and GLSL output, and an HLSL `cbuffer` with one reaches GLSL and SPIR-V output; all of them were refused. Array spellings are read outermost dimension first (`float[2][3]` is two `float[3]`). Metal output declares them with every extent, `float4 a[2][3]`, as it does structured-buffer structs that hold one; an element read takes all subscripts before its component selection or conversion (`params.a[i][j].xyz`), an HLSL array that shares its last register with the next member is read through the flattened index, and inner arrays are copied element by element.
- WGSL output declares a uniform array whose element stride is not a multiple of 16, such as CUDA or HIP `__constant__ float table[4];`, as `var<storage, read> table: array<f32, 4>;`, which keeps the packed stride the host writes with `cudaMemcpyToSymbol` or `hipMemcpyToSymbol`. Arrays of `float4`, of `float3` and of structures whose stride is a multiple of 16 stay `var<uniform>`, and an element whose layout is unknown is refused with `StructuredLayoutUnsupportedError` (`project.translate.structured-layout-unsupported`). It used to declare `var<uniform> table: array<f32, 4>;`, which naga rejects (`The array stride 4 is not a multiple of the required alignment 16`) (#3631).
- Slang output keeps declaration modifiers. `row_major` and `column_major` prefix matrix members of structs and constant buffers, `packoffset(c12)` follows its constant-buffer member, and `precise` stays on variables, so slangc lays out and computes them as it does the HLSL source. Matrix layout and precision qualifiers, `invariant` and `buffer_layout` no longer become semantics such as `: row_major`, which had replaced `SV_Position` on an `invariant gl_Position` output. A `row_major` struct member that holds a matrix is refused, because Slang only accepts the qualifier on a matrix.
- GLSL output declares a global constant that reads a specialization constant, such as `const int doubled = count * 2;` or `const int sizes[2] = int[2](count, 4);`, after the specialization constant, where it used to come first and `glslang` refused the undeclared name. When the shader is compiled to SPIR-V (`GL_SPIRV` or `VULKAN` defined), a constant that SPIR-V cannot build as a specialization-constant operation, such as an array, structure, matrix or floating-point value, is declared as a global variable with the same initializer, because `glslang` refuses it as a constant ("global const initializers must be constant"); GLSL ES output then enables `GL_EXT_shader_non_constant_global_initializers` in that branch, before its precision statements. Integer and Boolean scalar and vector constants stay constants.
- GLSL output writes `imageSamples(img)` for a multisample storage image such as `image2DMS` or `uimage2DMSArray`, and keeps `textureSamples(s)` for a multisample sampler, so GLSL input that queries an image's sample count also survives the round trip. It used to write `textureSamples(img)`, which GLSL defines only for samplers and glslang rejects (`'textureSamples' : no matching overloaded function found`) (#2460, #2457).
- Metal output passes an array argument that does not live in thread storage to a by-value array parameter as a thread copy, `pick(crosstl_array_value<float4, 2>{{u.v[0], u.v[1]}}.v)`. Constant-buffer members, program-scope constants, threadgroup and buffer arrays and array literals had failed with "no matching function for call", because the parameter is a `thread` pointer, and the elements now pass through the constant-buffer layout conversions, such as the `float4` slots of a std140 `float[3]` and the register words of a row-major matrix. An argument with effects, which the copy would evaluate once per element, is refused.
- Metal output passes an `inout` or `out` argument through a thread copy when another access during the call would alias the `thread T&` it binds: another argument writes the variable (`scale(m, bump(m))`), a second reference parameter binds it (`h(m, m)`), or the callee reads or writes it as a module-scope variable. A call whose enclosing call reaches a copied argument through an earlier argument (`atomicAdd(tile[i], bump(tile[i]))`) now stores its copy back before the enclosing call runs. Such a copy inside a conditional expression is refused with `MetalLoweringRefused`. Each case used to read or keep a value that GLSL's copy-in and copy-out rules rule out.
- Metal output evaluates a call that copies `inout` or `out` arguments in place when it sits in an operand that is evaluated conditionally: the right operand of `&&` or `||`, or an arm of `?:`, including `while` and `if` conditions. Its copies are made and stored back only when the operand runs. Such a call used to be refused.
- Metal output evaluates a call that copies `inout` or `out` arguments in a `do`/`while` condition or a `for` update in place, with its temporaries declared before the loop, so its copies are made and stored back on every evaluation, including one a `continue` reaches. Both headers used to be refused with an untyped `ValueError`. A `switch` selector of a type other than `int` or `uint` that copies arguments is refused with `MetalLoweringRefused`.
- A one-value brace initializer of a plain enumeration, such as `Mode m = {Mode::B};` or `enum class Mode { A, B }; Mode m = {Mode::B};` in CUDA input, is its value on every target, as if `Mode m = Mode::B;` were written, and so is a nested list such as `{{Mode::B}}`. A list with more than one value, such as `Mode m = {Mode::A, Mode::B};`, is refused with `ScalarBraceInitializerError` (`project.translate.scalar-brace-initializer-invalid`). WGSL output used to write `var m: i32 = array(Mode_B);`, which naga rejects, SPIR-V output a composite of `int`, which spirv-val rejects, Rust output `let m: Mode = [Mode::B];`, which rustc rejects, and Mojo output `var m: Mode = InlineArray[Mode, 1](Mode_B)`; GLSL output refused the program with `OpenGLAggregateInitializerError` (#4190).
- HLSL and Metal output read module-scope stage inputs (`in vec3 pos @location(0);` in CrossGL written the GLSL way) from the entry point's input structure. HLSL had declared them as globals, which are uniforms, and Metal as `constant` values initialized to zero. A program whose module-scope inputs could belong to more than one vertex or fragment stage is refused, as module-scope outputs already were.
- SPIR-V input declares a named `OpSpecConstantComposite` array, such as an `int[3]` built from a specialization constant, as `const int arr[3] = {SPEC_I, 2, 3};`. It used to leave the declaration out while the function body still read `arr`. A named specialization constant whose type CrossGL cannot name is refused with `SpirvInstructionUnsupportedError` instead of being dropped (#4204).
- Metal output passes more arrays to by-value array parameters. An array member of a GLSL buffer block held as `device uchar*` is a thread copy read through the block's byte offsets; the emitter had printed `outp.v`, a member access on a pointer, which the Metal compiler rejected. A `const` local array passed to a callee that writes its copy is a thread copy too, where the call was refused. The `metal::array<T, N>` result of a user function is passed as `f(...).data()`, which the Metal compiler had rejected; a nested array result is refused.
- Rust output value-initializes the elements of an array of a plain enumeration that a brace list leaves out, as in `M arr[2] = {};` or `M arr[2] = { {} };`, to the variant whose discriminant is 0, as `M m = {};` is: `let arr: [M; 2] = [M::B, M::B];` for `enum M { A = 2, B = 0 }`, and `[M::B; 3]` for a left-out row of `M grid[2][3]`. An enumeration with no zero-valued variant, or whose variants carry data, is refused with `RustEnumEmptyBraceError` (`project.translate.rust-enum-empty-brace-unsupported`). It used to fill them with `Default::default()`, the first variant (`M::A`, whose value is 2), and a shader-scope array such as `M table[2] = {};` became a `static` that rustc rejects (E0015) (#4189).
- HLSL output passes array values to by-value array parameters. An array literal argument is built as the array member of a constructed value struct (`crossgl_array_value_float_4(a, b, c, d).v`), and a `T[N]` call result, held as a vector or a `CrossGLFixedArray` struct, or a local initialized from one is flat-cast to the parameter's array type through a typedef. Literals were refused, and lowered values were passed unchanged, which DXC rejected. Struct constructor fields of array type take both forms as well. A lowered value passed to an `out` or `inout` array parameter is refused.
- Metal output keeps the array parameters of a function declared with a prototype, such as `buffer uint values[]` in `void put(buffer uint values[], uint i);`. The prototype's missing body made every array parameter look unused, so both the prototype and the definition lost it and its call sites lost the argument. Metal output also drops an unused array parameter only when every overload with that name leaves it unused, so `f(float a[2], float x)` no longer removes `x` from `f(float x, float y)`.
- Metal output passes a shader-scope buffer, texture or sampler to the specialization of a generic function that uses it, and to every helper that calls that specialization: `T twice<T>(T x) { return x + T(outp[0]); }` becomes `float twice_float(float x, device float *outp)`, called as `twice_float(v, outp)`. A non-generic overload with the same name no longer takes the generic one's buffers. It used to leave the parameter out of the specialization, which the Metal compiler rejects with `use of undeclared identifier 'outp'` (#3824).
- Metal output passes a constant buffer to the specialization of a generic function that reads one of its members, and to every helper that calls that specialization: `T scaled<T>(T v) { return v * T(gain); }` becomes `float scaled_float(float v, constant Params& params)`, called as `scaled_float(1.0, params)`. It used to leave the parameter out of the specialization, which the Metal compiler rejects with `use of undeclared identifier 'params'` (#4269).
- Slang output keeps the memory qualifiers of a resource: `volatile`, `restrict` and `writeonly` on storage images and buffers, `readonly` on a storage image, MSL's `coherent(device)`, and every qualifier of a GLSL storage block lowered to a structured or byte-address buffer, which lost even `coherent`. slangc decorates them `Volatile`, `Restrict`, `NonReadable`, `NonWritable` and `Coherent` in SPIR-V. What slangc cannot declare is refused with a typed diagnostic instead of dropped: coherence at system scope, a coherent or volatile parameter, and a coherent or volatile resource passed to a function (slangc E31201 and E30048). `readonly` and `writeonly` are left off a resource passed to a function, which slangc would otherwise reject.
- HLSL and GLSL output refuse a constant buffer that holds a struct whose members carry `@offset` or `@align` placing them where the buffer's layout would not, with `MemberPlacementError` (`project.translate.member-placement-unsupported`). They used to place the struct's members by the layout alone, and HLSL output wrote the attribute as a semantic named `offset`. Placements that agree with the layout are accepted.
- HLSL output reads a dynamically indexed row of a matrix with more rows than columns (`float3x2`, `float4x2`, `float4x3`) from a constant buffer, an array element or a structure member through a parameter copy, `crossgl_matrix_value_float3x2(m[k])[r]`. DXC lowered such a row through a temporary sized by the column count, which DXIL validation rejected as an out-of-bounds access. Written rows keep the direct form.
- A `row_major` uniform block, or block member, of structure type now reaches the matrices the structure holds in HLSL, Slang, Metal and SPIR-V output. The member takes a row-major copy of its structure. HLSL had written `row_major L l;`, which DXC rejects, Metal read each nested matrix transposed, and SPIR-V and Slang refused the member. SPIR-V output also lays out and decorates a structure's own `row_major` matrix members. Reading such a member as a whole structure is refused. GLSL output keeps `layout(row_major)` on the member. Buffer blocks get the same copies, whose matrices are stored transposed, so their nested row-major matrices no longer read column-major in any of the four outputs.
- Slang output translates a matrix element whose subscripts are operators, conditionals or scalar conversions of names, such as `m[k][(i / 4u) % 2u][i % 3u]`. It refused them as subscripts it could not reorder, although they write nothing; subscripts with increments are still refused.
- GLSL input keeps the `coherent`, `volatile`, `restrict` and `writeonly` qualifiers, and the Vulkan memory-model qualifiers such as `workgroupcoherent`, of a named storage block whose only member is a runtime array, such as `layout(std430, binding = 1) coherent buffer B1 { float d[]; } b1;`. It imports as `RWStructuredBuffer<float> b1 @binding(1) @coherent;`, where it used to drop every qualifier but `readonly`, so GLSL, SPIR-V and Metal output lost the memory semantics and HLSL output did not refuse a `volatile` block.
- Metal input reads `read`, `write` and `read_write` as names wherever a name belongs: a function such as `uint read(Wrapper value)`, at global or namespace scope, as a prototype or an overload, a call such as `wrappers::read(x)`, a parameter, a local or a structure member. `texture2d<float, access::read>` keeps its access mode, and `tex.read(gid)` is still a texture read. It used to raise `MetalParserError` (`Expected SEMICOLON, got LBRACE` for such a function, `Unexpected token in expression: READ` for such a parameter) (#2144).
- HLSL input places the members of a `[[vk::push_constant]]` struct where DXC's default relaxed layout puts them, so `struct C { float s; float3 v; };` keeps `v` at byte 4 in SPIR-V and HLSL output instead of moving it to byte 16, and keeps every member's `[[vk::offset(N)]]` as `@offset(N)`. SPIR-V and HLSL output accept a vector at such a relaxed offset (Vulkan 1.3 section 15.6.4), and DirectX, Metal and Slang output no longer write `@offset` or `@align` as a semantic.
- Slang output lowers `length()` of an array member of a std140 or std430 buffer block. A runtime array counts the whole elements that the bound buffer holds after its offset, `int((cgl_buffer_bytes(b) - offset) / stride)` through `GetDimensions`, as in DirectX output, and a sized array gives its size. Slang output used to print `block.member.length()` on the byte-address buffer unchanged, which slangc refused.
- Metal output keeps the coherence and `volatile` of buffers passed to helpers and of GLSL storage blocks. A helper parameter that receives a coherent or volatile buffer now carries the widest coherence and every `volatile` among its callers' arguments, through any chain of helpers, so `xcrun metal` no longer rejects the call ("would lose coherent(device) qualifier"); a coherent array parameter is written as the pointer it decays to, because MSL refuses `coherent` on array syntax. A GLSL `coherent` or `volatile` buffer block lowered to `device uchar*` keeps the qualifier on the parameter and on every byte-offset load, store, atomic, aggregate-load helper and compare-exchange helper; before, `reinterpret_cast<device T*>` dropped it and compiled to plain loads and stores.
- CUDA output declares a GLSL `readonly buffer` block, a sized array and a scalar as HIP output does, such as `__device__ B0 b0;` for `readonly buffer B0 { float fin[]; };`, and keeps `const` only on the element type of a runtime-sized array, such as `__device__ const float* values;`. It used to declare them as uninitialized `__device__ const` objects, which CUDA compilers reject; assignments and atomics on a readonly block are still refused (#4192).
- GLSL output keeps `layout(row_major)` on a shader storage block member of structure type, such as `layout(row_major) L ls[64];`, which makes the structure's matrices row-major at any depth (GLSL 4.60 section 4.4.5). Uniform blocks kept it but storage blocks dropped it, so a GLSL storage block with `layout(std430, row_major)` and nested structures read their matrices column by column after translation.
- Metal output keeps the coherence of storage images. A `coherent` or `globallycoherent` image becomes `texture2d<T, access::A, memory_coherence::memory_coherence_device>` (MSL 3.2), and so does every helper parameter that receives it, because the coherence argument is part of the texture type. Before, the qualifier was dropped and the texture kept MSL's default threadgroup coherence. A `volatile` image and one that is coherent at system scope are refused, because MSL textures have neither.
- Metal output copies a C array whose value steps a variable, such as `part = b[i++];`, through a thread reference bound once, `thread auto &src = b[cgl_rule_wrap_postinc(i)];`, and then copies element by element from `src`. It used to copy through `b[cgl_rule_wrap_postinc(i)][k]`, stepping `i` once per element. A stepped copy from storage other than `thread` is refused with `MetalLoweringRefused`. The `x++`/`--x` helpers are no longer counted as free of side effects, so every lowering that repeats an operand now sees them.
- HLSL, Slang and Metal output lower arrays of arrays in std140 and std430 buffer blocks, such as `float grid[2][3]` or a runtime `ivec2 rows[][2]`, at and inside struct members, where the whole block used to be refused. Each dimension's stride is the next dimension's extent times its stride, so `grid[i][j]` reads `i * 12 + j * 4` bytes past `grid` in std430. A block array read or written whole (`float c[4] = data.row;` or `data.grid[1] = r;`) is refused with `BufferBlockUnsupportedError` (`project.translate.<target>-buffer-block-unsupported`), where HLSL and Slang output used to read or write only its first element and Metal output printed it unchanged; Metal output still copies a sub-array assignment element by element. HLSL and Slang output also store a component written to a vector element of a block array, `data.vecs[1].y = s;`, where they used to assign to a load.
- HLSL, Slang and Metal output read and write whole arrays of std140 and std430 buffer blocks: `float c[4] = data.row;`, `f(data.row)`, `part = data.grid[i];` and `data.grid[1] = r;`, including arrays of arrays, vectors and structs. HLSL and Slang build the value as `crossgl_array_value_float_4(e0, e1, e2, e3).v` from the element loads and write through a temporary array and one store per element. Metal initializes from a brace list of the element loads and copies assignments element by element. These uses were refused. A whole-array access whose subscripts have side effects is still refused.
- HIP and OpenCL input read a parameter, cast or local declaration whose type a line break splits after its base type, a `*` or `&`, a postfix qualifier such as `restrict` or `const`, or `pipe`, as they read it on one line: `__global float`, a line break and `* in` imports as `@group(0) @binding(0) var<storage, read_write> in_: array<f32>`, and `float widen(unsigned`, a line break and `v)` as `f32 widen(u32 v)`. They used to raise `SyntaxError` (`Expected RPAREN, got STAR` for a parameter, `Unexpected token in expression` for a function or a cast) (#3753).
- SPIR-V output declares a row-major array of non-square matrices in a uniform block, such as `layout(row_major) mat3x2 k[2]`, with its own array type carrying the row-major `ArrayStride` (32 bytes for `mat3x2` in std140, where a column-major array of the same type takes 48), as glslang does, at every level of a nested array. It had refused the member, because the array type was shared with column-major uses.
- CrossGL reads `a < b && b > (c)` and `i < n && n > (i + 1)` as two comparisons joined by `&&` (or `||`), not as a generic call `a<b && b>(c)`: a generic argument list in an expression takes `&&` and `||` only inside parentheses, as in `f<(a && b)>(x)`. Rust output used to write `a:: < b && b > (c)`, which rustc rejects, and SPIR-V output refused the program with `SPIRVBuiltinLoweringError` (`Cannot lower call to 'a<b && b>'`) (#3674).
- GLSL `GL_KHR_memory_scope_semantics` coherence (`devicecoherent`, `queuefamilycoherent`, `workgroupcoherent`, `subgroupcoherent`, `shadercallcoherent` and `nonprivate`) on storage buffers and images reaches every output that used to drop it. HLSL output writes `globallycoherent`, where it used to write the qualifier as an array size (`RWStructuredBuffer<uint> b[workgroupcoherent]`); SPIR-V output decorates the resource `Coherent`; GLSL output keeps the scoped qualifier under `#pragma use_vulkan_memory_model` and writes `coherent` otherwise, since glslang ignores the scoped names without the pragma; Metal output writes `coherent(threadgroup)` on a buffer coherent at workgroup or subgroup scope and `coherent(device)` on the others. GLSL input reads `devicecoherent`, and the CrossGL parser reads a scoped qualifier before `buffer` in a `layout(...) workgroupcoherent buffer B { ... }` block (#4252).
- HIP input divides a surface call's x byte offset by the element size, folding `x * 16` on a `float4` surface to texel `x` and `x * 4 + 4` on a `float` surface to texel `x + 1`, and refuses offsets it cannot prove are multiples of the element size, with `HipSurfaceByteOffsetError` (`project.translate.hip-surface-byte-offset-unproven`). The element type comes from the template argument, from the destination of a pointer read, or from the write value, including member and array accesses, and `x * sizeof(T)` divides for a template parameter `T` whatever its size. It used to pass offsets other than `x * sizeof(T)` and `x * N` through unchanged, so `surf2Dread(&p.c, img, x * 16, y)` read texel `16x` and `surf2Dwrite(p.c, img, x * 16, y)` wrote texel `16x` (#4197).
- HLSL, Slang and Metal output store a write to a column of a matrix in a std140 or std430 buffer block (`data.m[c] = v;`, `data.m[c].y = s;`, `data.m[c][r] += s;`) into that column's bytes. They used to assign to a loaded copy of the matrix: DXC and slangc refused the assignment, and the Metal compiler accepted it and dropped the write.
- Slang output reads and writes a matrix in a storage buffer column by column. A loaded `mat3` was the transpose of the stored matrix, a non-square matrix such as `mat2x3` did not compile, and a store wrote rows into the buffer's columns; loads now build `transpose(float2x3(column0, column1))` and stores select each column, as the HLSL output does.
- GLSL output passes a shared array to a by-value array parameter, as in `float bumps(float a[64], uint i)` called as `bumps(s, i)` on `shared float s[64];` or an HLSL `groupshared` array, and copies it whole, as in `float t[64] = s;` or `u = s;`. GLSL copies the array in (GLSL 4.60 section 6.1.1). The generator had refused each of these with "OpenGL cannot emit a workgroup pointer as a first-class value". A Metal `threadgroup` pointer named bare is still refused.
- GLSL input marks an array parameter without `out` or `inout` as `in`, as HLSL input does, because GLSL passes it by value. CUDA and HIP output therefore copy such an array when the function writes it: `float first(float v[2]) { v[0] = 3.0; ... }` used to change the caller's array through the decayed pointer. (#3162)
- GLSL, HLSL, Metal, SPIR-V and Slang output apply a matrix order written on a block. `@row_major` on a `cbuffer`, or on a block instance such as `B b @row_major @glsl_buffer_block(std430)`, now orders every matrix member that names no order of its own, including arrays of matrices and structs holding them, as GLSL's block-level `layout(row_major)` does. The generators had read matrix orders from members only, so such a block was declared column-major and each matrix read transposed. A struct that blocks declare with different orders is refused.
- Slang output declares a function prototype without a body, which slangc had rejected as a second definition, and reads a structured buffer's element count with `GetDimensions(count, stride)`, the only overload slangc provides. SPIR-V output inlines the definition of a helper that takes a storage buffer instead of also counting its prototype as an overload.
- HLSL and Slang output of `buffer_dimensions(b, count)` on a structured buffer assign the element count (`count = cgl_buffer_count(b);`) instead of calling `b.GetDimensions(count)`, which has no overload for a structured buffer and was rejected by DXC and slangc; output lists no `GetDimensions` overload takes, such as a stride for a byte-address buffer, are refused with a diagnostic.
- Slang output translates GLSL geometry shaders. It printed the GLSL form, with `gl_in`, `EmitVertex()` and stage outputs that Slang reads as uniforms, so slangc rejected it; the stage now takes the stream form the HLSL output uses, with `[instance(N)]` for `invocations = N`. Writes to `gl_Layer`, `gl_ViewportIndex` and `gl_PrimitiveID` now reach their system values in Slang. A stage with more than one vertex stream stays refused, because slangc 2026.20 aborts on a second stream object.
- HIP, Metal, HLSL and GLSL output write a generic struct member of type `vecN<T>`, such as `vec2<T> v;` in `struct VBox<T>`, with its concrete vector once the struct is specialized: `float2 v;` in HIP, Metal and HLSL output and `vec2 v;` in GLSL output, including an array of generic vectors such as `vec2<T> q[2];`. They used to write `T2 v;`, which the HIP, Metal and HLSL compilers reject with `unknown type name 'T2'` and which GLSL output refused outright (#4198).
- Slang output moves three-component byte-address values through helpers that load or store two components and then one, `__crossgl_load3_uint(buffer, offset)` and `__crossgl_store3_uint(buffer, offset, value)` (and typed 16- and 64-bit forms). slangc 2026.18.3 compiles `Load3`, `Store3`, `Load<T3>` and `Store<T3>` at a constant offset that is a nonzero multiple of the vector's size to the wrong bytes in SPIR-V (`Store3(48, v)` writes bytes 64..75), which a std430 `vec3` at byte 48 or a `mat3` column at byte 96 hits. Slang output also lowers `buffer_load2`..`buffer_load4` and `buffer_store2`..`buffer_store4`, which HLSL `LoadN`/`StoreN` calls import as, where it used to print them unchanged.

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

- HLSL import translates template value parameters (`template <typename T, uint N>`) on function and struct templates, including templates inside namespaces, to CrossGL generic parameters bound at each use (`sum<uint, 5u>`, `Ring<float, 4u>`, with defaults of struct templates spelled out). A defaulted value parameter of a function template is still refused with `DirectXImportRefused`.
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
