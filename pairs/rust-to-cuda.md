# CrossTL translation matrix: Rust to CUDA

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Rust probe whose feature CUDA has too straight to CUDA, and nvcc checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 81.0 %: 204 supported, 28 refused, 20 wrong, 1 untested.

## Gaps

<details>
<summary>48 gaps: 28 refused, 20 wrong</summary>

Wrong, by cause:

- `expression must have class type but it has type "float"` (6): `Float::acosh`, `Float::asinh`, `Float::atanh`, `Float::cbrt`, `Float::exp_m1`, `Float::ln_1p`
- `name followed by "::" must be a class or namespace name` (5): `Mat3::from_angle`, `Mat4::abs`, `Mat4::from_rotation_x`, `Mat4::from_rotation_z`, `Vec2::from_angle`
- `expression must have class type but it has type "uint" (aka "unsigned int")` (4): `unsigned-buffer`, `unsigned-float-conversion`, `unsigned-negation`, `unsigned-wraparound`
- `"t" has already been declared in the current scope`: `sibling-redeclaration`
- `class "float2" has no member "to_angle"`: `Vec2::to_angle`
- `class "float3" has no member "copysign"`: `Vec3::copysign`
- `expression must have class type but it has type "unsigned int"`: `unsigned-countdown`
- `no suitable conversion function from "float4" to "float" exists`: `read-write-image`

Refused, by diagnostic:

- `project.translate.crossgl-function-body-parse-failed: CrossGLFunctionBodyParseError: CrossGL could not parse the body of function '<probe>' at AS 'as': Expected RBRACE, got AS 'as'` (8): `atomic_and`, `atomic_compare_exchange`, `atomic_exchange`, `atomic_i_add`, `atomic_or`, `atomic_u_max`, `atomic_u_min`, `atomic_xor`
- `project.translate.cuda-lowering-unsupported: CudaLoweringRefused: CUDA cannot lower N construct: function: call to 'GroupMemoryBarrierWithGroupSync', a built-in function of another shading language that CUDA does not define and the program does not declare (target cuda) [project.translate.cuda-lowering-unsupported]` (6): `barrier-in-loop`, `shared-2d-array`, `shared-memory`, `shared-memory-in-helper`, `shared-struct-array`, `shared-vector-array`
- `project.translate.cuda-tuple-field: TupleFieldUnsupportedError: tuple field access '.N' is not supported: no cuda lowering exists for tuple values (target cuda) [project.translate.cuda-tuple-field]` (2): `Float::sin_cos`, `Vec3::sin_cos`
- `project.translate.rust-import-unsupported: RustImportRefused: CrossGL cannot represent N Rust construct: inline assembly 'asm!' is written for one instruction set, and CrossGL has no inline assembly (target cgl) [project.translate.rust-import-unsupported]` (2): `texel-buffer`, `texture-1d-array`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'acc' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target cuda) [project.translate.undeclared-identifier]` (2): `switch-case-scope`, `switch-default-position`
- `project.translate.crossgl-syntax-error: CrossGLSyntaxError: Expected a structure name after 'struct', got VAR 'var'`: `reserved-type-name`
- `project.translate.cuda-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array '<probe>' is a parameter: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last member of a storage blo`: `buffer-in-helper`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier '<probe>' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target cuda) [project.translate.undeclared-identifier]`: `unsigned-switch`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'output' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target cuda) [project.translate.undeclared-identifier]`: `reserved-global-name`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier 'x' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a CrossGL built-in value (target cuda) [project.translate.undeclared-identifier]`: `deferred-initialization`
- `project.translate.undeclared-identifier: UndeclaredIdentifierError: identifier '{ auto _rust_expr_value_1; j += N; _rust_expr_value_1 =(j > N); return _rust_expr_value_1; }' in function '<probe>' is not declared: it is neither a local, parameter, stage or module variable, constant, constant-buffer member, enumerator, function or generic parameter of the program nor a`: `condition-assignment`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'Bits' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target cuda) [project.translate.undeclared-type]`: `union`
- `project.translate.undeclared-type: UndeclaredTypeError: type 'T' is not declared: it is neither a CrossGL built-in type nor a structure, enumeration, type alias or generic parameter of the program (target cuda) [project.translate.undeclared-type]`: `generic-function`

</details>

<details>
<summary>1 untested: function 1</summary>

- `untested in the rust input row: the probe itself is rejected: error: Hypot not supported yet`: `Float::hypot`

</details>
