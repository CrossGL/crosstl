# CrossTL translation matrix: Rust to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Rust probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 68.9 %: 173 supported, 34 refused, 44 wrong, 1 untested.

## Gaps

<details>
<summary>78 gaps: 34 refused, 44 wrong</summary>

Wrong, by cause:

- `invalid call to '__iadd__': value passed to 'rhs' cannot be converted from 'Float32' to 'Float64'` (8): `block-scope`, `break`, `continue`, `for-loop`, `for-loop-variants`, `nested-loop`, `unsigned-countdown`, `value-parameter-mutation`
- `unexpected token in expression` (4): `Float::nan`, `Mat4::abs`, `Mat4::div_scalar`, `Mat4::recip`
- `'UInt32' value has no attribute 'wrapping_add'` (2): `unsigned-float-conversion`, `unsigned-wraparound`
- `invalid call to '__add__': value passed to 'rhs' cannot be converted from 'Float32' to 'Float64'` (2): `infinite-loop`, `loop-condition-call`
- `invalid call to '__lt__': value passed to 'rhs' cannot be converted from 'Int32' to 'Int'` (2): `empty-statement`, `while-loop`
- `ValueError: invalid literal for int() with base N: 'N'`: `constant-array-size`
- `'Float32' value has no attribute 'acosh'`: `Float::acosh`
- `'Float32' value has no attribute 'asinh'`: `Float::asinh`
- `'Float32' value has no attribute 'atanh'`: `Float::atanh`
- `'Float32' value has no attribute 'cbrt'`: `Float::cbrt`
- `'Float32' value has no attribute 'exp_m1'`: `Float::exp_m1`
- `'Float32' value has no attribute 'ln_1p'`: `Float::ln_1p`
- `'SIMD[.bool, N]' value has no attribute 'all'`: `bool-vector-logic`
- `'SIMD[.float32, N]' value has no attribute 'cmplt'`: `vector-comparison`
- `'SIMD[.float32, N]' value has no attribute 'copysign'`: `Vec3::copysign`
- `'SIMD[.float32, N]' value has no attribute 'project'`: `Vec4::project`
- `'SIMD[.float32, N]' value has no attribute 'saturate'`: `Vec3::saturate`
- `'SIMD[.float32, N]' value has no attribute 'to_angle'`: `Vec2::to_angle`
- `'SIMD[.int32, N]' value has no attribute 'as_uvec2'`: `vector-shift`
- `'SIMD[.int32, N]' value has no attribute 'as_vec2'`: `vector-conversion`
- `'SIMD[.int32, N]' value has no attribute 'cmplt'`: `integer-vector-comparison`
- `'SIMD[.int32, N]' value has no attribute 'cmpne'`: `bool-vector-conversion`
- `'SIMD[.uint32, N]' value has no attribute 'cmpne'`: `uniform-bool-vector`
- `'SIMD[.uint32, N]' value has no attribute 'wrapping_mul'`: `unsigned-vector-arithmetic`
- `'UInt32' value has no attribute 'wrapping_neg'`: `unsigned-negation`
- `'UInt32' value has no attribute 'wrapping_sub'`: `unsigned-buffer`
- `cannot implicitly convert 'Float64' value to 'Float32'`: `aggregate-equality`
- `invalid call to '__eq__': value passed to 'rhs' cannot be converted from 'Int32' to 'Int'`: `loop-counter-write`
- `invalid call to 'outer': l-value of type 'Float64' cannot be converted to reference of type 'Float32'`: `out-parameter-forwarding`
- `invalid redefinition of 't'`: `sibling-redeclaration`
- `no matching function in call to 'power'`: `Float::powi`

Refused, by diagnostic:

- `project.translate.undeclared-function: UndeclaredFunctionError: function '<feature>' is not declared: it is neither a CrossGL built-in function nor a function, structure method, type or callable value of the program (target mojo) [project.translate.undeclared-function]` (8): `subgroup_f_add`, `subgroup_f_max`, `subgroup_f_min`, `subgroup_inclusive_f_add`, `subgroup_shuffle`, `subgroup_shuffle_down`, `subgroup_shuffle_up`, `subgroup_shuffle_xor`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: function: call to 'GroupMemoryBarrierWithGroupSync', a built-in function of another shading language that Mojo does not define and the program does not declare (target mojo) [project.translate.mojo-lowering-unsupported]` (7): `barrier-in-loop`, `shared-2d-array`, `shared-memory`, `shared-memory-in-helper`, `shared-struct-array`, `shared-vector-array`, `workgroup_memory_barrier_with_group_sync`
- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at AS 'as': Expected RBRACE, got AS 'as'` (5): `atomic_i_add`, `atomic_load`, `atomic_store`, `atomic_u_max`, `atomic_u_min`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: type: the built-in type 'auto' has no Mojo spelling (target mojo) [project.translate.mojo-lowering-unsupported]` (2): `multi-variable-loop`, `multiple-declarators`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter groups has no GPU index source for semantic <feature>; a Mojo kernel's arguments are the buffers and uniform values the host passes` (2): `num-subgroups`, `num-workgroups`
- `project.translate.crossgl-syntax-error: CrossGLSyntaxError: Expected a structure name after 'struct', got VAR 'var'`: `reserved-type-name`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter group has no GPU index source for semantic gl_SubgroupID; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter index has no GPU index source for semantic gl_LocalInvocationIndex; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `local-invocation-index`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lane has no GPU index source for semantic gl_SubgroupInvocationID; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-invocation-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter width has no GPU index source for semantic gl_SubgroupSize; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-size`
- `project.translate.mojo-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array '<probe>' is a parameter: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of a storage blo`: `buffer-in-helper`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'output' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target mojo) [project.translate.undeclared-identifier]`: `reserved-global-name`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'x' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target mojo) [project.translate.undeclared-identifier]`: `deferred-initialization`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier '{ auto _rust_expr_value_1; j += N; _rust_expr_value_1 =(j > N); return _rust_expr_value_1; }' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a`: `condition-assignment`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'T' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target mojo) [project.translate.undeclared-type]`: `generic-function`

</details>

<details>
<summary>1 untested: function 1</summary>

- `untested in the rust input row: the probe itself is rejected: error: Hypot not supported yet`: `Float::hypot`

</details>
