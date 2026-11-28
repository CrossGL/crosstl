# CrossTL translation matrix: Rust to HIP

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Rust probe whose feature HIP has too straight to HIP, and hipcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 80.0 %: 204 supported, 30 refused, 21 wrong, 1 untested.

## Gaps

<details>
<summary>51 gaps: 30 refused, 21 wrong</summary>

Wrong, by cause:

- `member reference base type 'float' is not a structure or union` (6): `Float::acosh`, `Float::asinh`, `Float::atanh`, `Float::cbrt`, `Float::exp_m1`, `Float::ln_1p`
- `member reference base type 'unsigned int' is not a structure or union` (5): `unsigned-buffer`, `unsigned-countdown`, `unsigned-float-conversion`, `unsigned-negation`, `unsigned-wraparound`
- `use of undeclared identifier 'Mat4'` (3): `Mat4::abs`, `Mat4::from_rotation_x`, `Mat4::from_rotation_z`
- `no member named 'copysign' in 'HIP_vector_type<float, 3>'`: `Vec3::copysign`
- `no member named 'to_angle' in 'HIP_vector_type<float, 2>'`: `Vec2::to_angle`
- `no viable conversion from 'HIP_vector_type<float, 4>' to 'float'`: `read-write-image`
- `redefinition of 't' with a different type: 'float' vs 'int'`: `sibling-redeclaration`
- `use of undeclared identifier 'Mat3'`: `Mat3::from_angle`
- `use of undeclared identifier 'Vec2'`: `Vec2::from_angle`
- `use of undeclared identifier 'f32'`: `Float::nan`

Refused, by diagnostic:

- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at AS 'as': Expected RBRACE, got AS 'as'` (9): `atomic_and`, `atomic_compare_exchange`, `atomic_exchange`, `atomic_i_add`, `atomic_i_sub`, `atomic_or`, `atomic_u_max`, `atomic_u_min`, `atomic_xor`
- `project.translate.hip-lowering-unsupported: HipLoweringRefused: HIP cannot lower N construct: function: call to 'GroupMemoryBarrierWithGroupSync', a built-in function of another shading language that HIP does not define and the program does not declare (target hip) [project.translate.hip-lowering-unsupported]` (7): `barrier-in-loop`, `shared-2d-array`, `shared-memory`, `shared-memory-in-helper`, `shared-struct-array`, `shared-vector-array`, `workgroup_memory_barrier_with_group_sync`
- `project.translate.hip-tuple-field: TupleFieldUnsupportedError: tuple field access '.N' is not supported: no hip lowering exists for tuple values (target hip) [project.translate.hip-tuple-field]` (2): `Float::sin_cos`, `Vec3::sin_cos`
- `project.translate.rust-import-unsupported: RustImportRefused: CrossGL cannot represent N Rust construct: inline assembly 'asm!' is written for one instruction set, and CrossGL has no inline assembly (target cgl) [project.translate.rust-import-unsupported]` (2): `texel-buffer`, `texture-1d-array`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'acc' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target hip) [project.translate.undeclared-identifier]` (2): `switch-case-scope`, `switch-default-position`
- `project.translate.crossgl-syntax-error: CrossGLSyntaxError: Expected a structure name after 'struct', got VAR 'var'`: `reserved-type-name`
- `project.translate.hip-ast-node-unlowered: GeneratedAstTextError: the hip emitter has no lowering for an AST node and printed it instead of hip code: IdentifierNode(name=lambda)()) (target hip) [project.translate.hip-ast-node-unlowered]`: `condition-assignment`
- `project.translate.hip-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array '<probe>' is a parameter: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of a storage blo`: `buffer-in-helper`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier '<probe>' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target hip) [project.translate.undeclared-identifier]`: `unsigned-switch`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'output' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target hip) [project.translate.undeclared-identifier]`: `reserved-global-name`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'x' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target hip) [project.translate.undeclared-identifier]`: `deferred-initialization`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'Bits' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target hip) [project.translate.undeclared-type]`: `union`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'T' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target hip) [project.translate.undeclared-type]`: `generic-function`

</details>

<details>
<summary>1 untested: function 1</summary>

- `untested in the rust input row: the probe itself is rejected: error: Hypot not supported yet`: `Float::hypot`

</details>
