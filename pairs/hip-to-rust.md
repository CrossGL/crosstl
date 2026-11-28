# CrossTL translation matrix: HIP to Rust

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HIP probe whose feature Rust has too straight to Rust, and rust-gpu checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 70.7 %: 186 supported, 52 refused, 25 wrong, 2 untested.

## Gaps

<details>
<summary>77 gaps: 52 refused, 25 wrong</summary>

Wrong, by cause:

- `error[E0308]: mismatched types` (5): `constant-expression`, `constant-struct`, `constant-vector-array`, `struct-constructor`, `zero-initializer`
- `GLSL.std.N Exp: expected Result Type to be a N or N-bit scalar or vector float type` (2): `exp`, `expm1`
- `GLSL.std.N Log: expected Result Type to be a N or N-bit scalar or vector float type` (2): `log`, `log1p`
- `error[E0425]: cannot find type 'ptr' in this scope` (2): `buffer-in-helper`, `pointer-local`
- `GLSL.std.N Acos: expected Result Type to be a N or N-bit scalar or vector float type`: `acos`
- `GLSL.std.N Acosh: expected Result Type to be a N or N-bit scalar or vector float type`: `acosh`
- `GLSL.std.N Asin: expected Result Type to be a N or N-bit scalar or vector float type`: `asin`
- `GLSL.std.N Asinh: expected Result Type to be a N or N-bit scalar or vector float type`: `asinh`
- `GLSL.std.N Atan2: expected Result Type to be a N or N-bit scalar or vector float type`: `atan2`
- `GLSL.std.N Atan: expected Result Type to be a N or N-bit scalar or vector float type`: `atan`
- `GLSL.std.N Atanh: expected Result Type to be a N or N-bit scalar or vector float type`: `atanh`
- `GLSL.std.N Cosh: expected Result Type to be a N or N-bit scalar or vector float type`: `cosh`
- `GLSL.std.N Exp2: expected Result Type to be a N or N-bit scalar or vector float type`: `exp2`
- `GLSL.std.N Log2: expected Result Type to be a N or N-bit scalar or vector float type`: `log2`
- `GLSL.std.N Pow: expected Result Type to be a N or N-bit scalar or vector float type`: `pow`
- `GLSL.std.N Sinh: expected Result Type to be a N or N-bit scalar or vector float type`: `sinh`
- `GLSL.std.N Tan: expected Result Type to be a N or N-bit scalar or vector float type`: `tan`
- `GLSL.std.N Tanh: expected Result Type to be a N or N-bit scalar or vector float type`: `tanh`

Refused, by diagnostic:

- `project.translate.rust-refused: RustRefusal: Rust compute stage needs a workgroup size: rust-gpu takes it as literals` (12): `__syncthreads`, `barrier-in-loop`, `local-invocation-id`, `num-workgroups`, `shared-2d-array`, `shared-memory`, `shared-memory-in-helper`, `shared-struct-array`, `shared-vector-array`, `subgroup-invocation-id`, `subgroup-size`, `workgroup-id`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N constructs: stage: compute entry parameter 'tex' is not a compute built-in input; stage: compute entry parameter 'tex_sampler' is not a compute built-in input (target rust) [project.translate.rust-lowering-unsupported]` (9): `sampled-texture`, `texture-1d`, `texture-1d-array`, `texture-2d-array`, `texture-3d`, `texture-argument`, `texture-cube`, `texture-cube-array`, `texture-in-helper`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N constructs: texture argument: storage image parameter 'img' of '<probe>' receives no image with a format layout qualifier, and rust-gpu storage images without a format need the StorageImage*WithoutFormat capabilities; stage: compute entry parameter 'img' is not a compute built-in i` (7): `integer-storage-image`, `read-write-image`, `storage-image`, `storage-image-1d`, `storage-image-3d`, `storage-image-array`, `unorm-storage-image`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: '<feature>' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]` (5): `atomicSub`, `expm1f`, `log1pf`, `nan`, `nanf`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'params_block': type 'f32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]` (5): `uniform-cbuffer-vector`, `uniform-members`, `uniform-std140-array`, `uniform-std140-vector`, `uniform-vector-packing`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N constructs: built-in: '<feature>' is used by name, and the rust-gpu output binds no entry-point parameter for it; generation then failed: Rust compute stage needs a workgroup size: rust-gpu takes it as literals (target rust) [project.translate.rust-lowering-unsupported]` (3): `local-invocation-index`, `num-subgroups`, `subgroup-id`
- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_libm_sin on a N-bit floating-point operand has no helper for rust: the helper takes binary32 values only (target rust) [project.translate.rule-inexpressible]` (2): `sin`, `sincos`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'params_block': type 'u32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]` (2): `uniform-bool`, `uniform-unsigned`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'scale_block': type 'f32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]` (2): `uniform-buffer`, `uniform-in-helper`
- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_libm_cos on a N-bit floating-point operand has no helper for rust: the helper takes binary32 values only (target rust) [project.translate.rule-inexpressible]`: `cos`
- `project.translate.rule-inexpressible: RuleInexpressibleError: __cgl_round_away on a N-bit floating-point operand has no helper for rust: the helper takes binary32 values only (target rust) [project.translate.rule-inexpressible]`: `round`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: constant block 'params_block': type 'i32' has no uniform-block layout (target rust) [project.translate.rust-lowering-unsupported]`: `uniform-integer`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: compute entry parameter 'tex' is not a compute built-in input (target rust) [project.translate.rust-lowering-unsupported]`: `texel-buffer`
- `project.translate.rust-refused: RustRefusal: Rust struct 'Bits' has a @union_layout: the Rust generator has no union lowering, and a struct would give its overlapping members separate storage`: `union`

</details>

<details>
<summary>2 untested: function 2</summary>

- `the target's own probe is rejected: error: Hypot not supported yet` (2): `hypot`, `hypotf`

</details>
