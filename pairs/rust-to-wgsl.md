# CrossTL translation matrix: Rust to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Rust probe whose feature WGSL has too straight to WGSL, and naga checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 52.5 %: 196 supported, 96 refused, 81 wrong, 1 untested.

## Gaps

<details>
<summary>177 gaps: 96 refused, 81 wrong</summary>

Wrong, by cause:

- `Entry point compute_main at Compute is invalid` (45): `Image::fetch_with(sample_index)`, `Image::fetch_with_lod`, `Image::gather`, `Image::query_levels`, `Image::query_samples`, `Image::query_size_lod`, `Image::sample_by_gradient`, `Image::sample_by_lod`, `Image::sample_with(grad)`, `Image::sample_with(lod)`, `aggregate-equality`, `barrier-in-loop`, `descriptor-set`, `sampled-texture`, `shared-2d-array`, `shared-memory`, `shared-memory-in-helper`, `shared-sampler`, `shared-struct-array`, `shared-vector-array`, `specialization-constant`, `specialization-constant-in-helper`, `texture-1d`, `texture-2d-array`, `texture-3d`, `texture-argument`, `texture-cube`, `texture-cube-array`, `texture-in-helper`, `uniform-array`, `uniform-buffer`, `uniform-cbuffer-vector`, `uniform-in-helper`, `uniform-integer`, `uniform-integer-vector`, `uniform-members`, `uniform-std140-array`, `uniform-std140-vector`, `uniform-struct`, `uniform-struct-array`, `uniform-struct-copy`, `uniform-unsigned`, `uniform-vector`, `uniform-vector-array`, `uniform-vector-packing`
- `expected ';', found ":"` (15): `Mat4::abs`, `constant-matrix`, `f16_to_f32`, `f16x2_to_vec2`, `f32_to_f16`, `matrix-buffer`, `u16x2_to_vec2_snorm`, `u16x2_to_vec2_unorm`, `u8x4_to_vec4_snorm`, `u8x4_to_vec4_unorm`, `vec2_to_f16x2`, `vec2_to_u16x2_snorm`, `vec2_to_u16x2_unorm`, `vec4_to_u8x4_snorm`, `vec4_to_u8x4_unorm`
- `expected ';', found "("` (11): `BVec3::all`, `BVec3::any`, `Float::acosh`, `Float::asinh`, `Float::atanh`, `Float::cbrt`, `Image::fetch_with(lod)`, `Image::sample_depth_reference_with(bias)`, `Vec2::to_angle`, `Vec3::saturate`, `unsigned-buffer`
- `Entry point fragment_main at Fragment is invalid` (3): `Image::sample`, `Image::sample_bias`, `Image::sample_with(bias)`
- `expected ')', found ":"` (2): `matrix-array`, `struct-matrix-member`
- `expected ')', found "("`: `bool-vector-logic`
- `invalid field accessor 'y_axis'`: `uniform-matrix`
- `redefinition of 't'`: `sibling-redeclaration`
- `the type of 't' is expected to be 'array<f32, 4>', but got 'array<{AbstractFloat}, 1>'`: `constant-array-size`
- `the translation loses Decoration.Invariant`: `invariant-position`

Refused, by diagnostic:

- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target wgsl) [project.translate.undeclared-function]` (21): `subgroup_all`, `subgroup_and`, `subgroup_any`, `subgroup_ballot`, `subgroup_broadcast_first`, `subgroup_exclusive_f_add`, `subgroup_exclusive_f_mul`, `subgroup_f_add`, `subgroup_f_max`, `subgroup_f_min`, `subgroup_f_mul`, `subgroup_inclusive_f_add`, `subgroup_inclusive_f_mul`, `subgroup_or`, `subgroup_quad_broadcast`, `subgroup_quad_swap`, `subgroup_shuffle`, `subgroup_shuffle_down`, `subgroup_shuffle_up`, `subgroup_shuffle_xor`, `subgroup_xor`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable m declared as void` (14): `matrix-arithmetic`, `matrix-column-assignment`, `matrix-compound-assignment`, `matrix-from-vectors`, `matrix-indexing`, `matrix-negation`, `matrix-parameter`, `matrix-product`, `matrix-resize`, `matrix-vector-product`, `uniform-matrix2`, `uniform-matrix3`, `uniform-row-major-array`, `uniform-row-major-matrix`
- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at AS 'as': Expected RBRACE, got AS 'as'` (11): `atomic_and`, `atomic_compare_exchange`, `atomic_exchange`, `atomic_i_add`, `atomic_i_sub`, `atomic_load`, `atomic_or`, `atomic_s_max`, `atomic_s_min`, `atomic_store`, `atomic_xor`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable b declared as void` (5): `bool-vector-conversion`, `nested-unary`, `unsigned-negation`, `unsigned-vector-arithmetic`, `unsigned-wraparound`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image2D yet; split texture/sampler/storage bindings are required` (4): `Image::write`, `read-write-image`, `storage-image`, `unorm-storage-image`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable c declared as void` (3): `integer-vector-arithmetic`, `uniform-bool-vector`, `vector-truncation`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable p declared as void` (3): `bool-bitwise`, `bool-conversion`, `boolean-logic`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'acc' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target wgsl) [project.translate.undeclared-identifier]` (2): `switch-case-scope`, `switch-default-position`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable lt declared as void` (2): `integer-vector-comparison`, `vector-comparison`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL texture function textureCompareGrad yet` (2): `Image::sample_depth_reference_by_gradient`, `Image::sample_depth_reference_with(grad)`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target requires textureCompareLod() explicit sampler operand to be a comparison_sampler` (2): `Image::sample_depth_reference_by_lod`, `Image::sample_depth_reference_with(lod)`
- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @gl_ClipDistance on parameter 'clip' of '<probe>' is not a vertex input in GLSL; GLSL declares gl_ClipDistance as an input of fragment, geometry, tessellation_control, tessellation_evaluation and an output of geometry, mesh, tessellation_control, tessellation_evaluation, vertex (GLSL N.N section N.N)`: `clip-distance`
- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @gl_FragDepth on parameter 'depth' of '<probe>' is not a fragment input in GLSL; GLSL declares gl_FragDepth as an output of fragment (GLSL N.N section N.N) (target wgsl) [project.translate.builtin-value-unsupported]`: `fragment-depth`
- `project.translate.crossgl-syntax-error: CrossGLSyntaxError: Expected a structure name after 'struct', got VAR 'var'`: `reserved-type-name`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier '<probe>' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target wgsl) [project.translate.undeclared-identifier]`: `unsigned-switch`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'output' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target wgsl) [project.translate.undeclared-identifier]`: `reserved-global-name`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'x' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target wgsl) [project.translate.undeclared-identifier]`: `deferred-initialization`
- `project.translate.wgsl-foreign-builtin-call: WGSLForeignBuiltinCallError: WGSL target cannot lower the call to lambda(): lambda is a built-in function of another language, and WGSL declares no built-in function of that name (target wgsl) [project.translate.wgsl-foreign-builtin-call]`: `condition-assignment`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable arr declared as void`: `expression-swizzle`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable b0 declared as void`: `uniform-bool`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable d declared as void`: `diagonal-matrix`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable either declared as void`: `bool-array`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable f declared as void`: `vector-conversion`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable j declared as void`: `unsigned-countdown`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable l declared as void`: `vector-shift`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable r declared as void`: `operator-precedence`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable s declared as void`: `signed-division`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable t declared as void`: `buffer-row-major-matrix`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable t0 declared as void`: `buffer-row-major-array`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot infer a concrete type for local variable v declared as void`: `unsigned-float-conversion`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target cannot pass a part of '<probe>' to inout parameter x of grow: the argument is not a whole local variable, and a ptr<function, T> argument must be a whole function-scope variable (WGSL N.N.N); copy it into a local variable around the call`: `buffer-element-argument`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image1D yet; split texture/sampler/storage bindings are required`: `storage-image-1d`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image2DArray yet; split texture/sampler/storage bindings are required`: `storage-image-array`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type image3D yet; split texture/sampler/storage bindings are required`: `storage-image-3d`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type uimage2D yet; split texture/sampler/storage bindings are required`: `integer-storage-image`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target requires textureCompare() explicit sampler operand to be a comparison_sampler`: `Image::sample_depth_reference`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target supports texelFetch() calls with exactly N arguments; got N for texelFetch`: `Image::fetch`
- `project.translate.wgsl-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array '<probe>' is a parameter: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of a storage blo`: `buffer-in-helper`

</details>

<details>
<summary>1 untested: function 1</summary>

- `the target's own probe is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroup_elect`

</details>
