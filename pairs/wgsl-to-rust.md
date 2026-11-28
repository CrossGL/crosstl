# CrossTL translation matrix: WGSL to Rust

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature Rust has too straight to Rust, and rust-gpu checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 93.3 %: 321 supported, 23 refused, 0 wrong, 2 untested.

## Gaps

<details>
<summary>23 gaps: 23 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'float[N]' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]` (7): `array-return`, `barrier-in-loop`, `call-result-access`, `clip-distance`, `shared-memory`, `shared-memory-in-helper`, `struct-array-member`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'float[N][N]' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]` (2): `array-of-arrays`, `shared-2d-array`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'atomicSub' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]`: `atomicSub`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'Item[N]' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]`: `nested-member-assignment`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'Pair[N]' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]`: `shared-struct-array`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'buffer_dimensions' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]`: `runtime-array-length`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: function: 'vec4[N]' has no rust-gpu lowering for these operands (target rust) [project.translate.rust-lowering-unsupported]`: `shared-vector-array`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: uniform 'params': 'Params.b' is at std140 offset N, rust-gpu lays it out at offset N (target rust) [project.translate.rust-lowering-unsupported]`: `uniform-std140-vector`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: resource: uniform 'params': 'Params.m column N' is at std140 offset N, rust-gpu lays it out at offset N (target rust) [project.translate.rust-lowering-unsupported]`: `uniform-matrix3`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: built-in gl_SampleMaskIn is not a rust-gpu fragment input (target rust) [project.translate.rust-lowering-unsupported]`: `sample-mask-input`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: built-in gl_SampleMask is not a rust-gpu fragment output (target rust) [project.translate.rust-lowering-unsupported]`: `sample-mask`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: compute entry parameter 'group' is not a compute built-in input (target rust) [project.translate.rust-lowering-unsupported]`: `subgroup-id`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: compute entry parameter 'groups' is not a compute built-in input (target rust) [project.translate.rust-lowering-unsupported]`: `num-subgroups`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: compute entry parameter 'lane' is not a compute built-in input (target rust) [project.translate.rust-lowering-unsupported]`: `subgroup-invocation-id`
- `project.translate.rust-lowering-unsupported: RustLoweringRefused: Rust cannot lower N construct: stage: compute entry parameter 'width' is not a compute built-in input (target rust) [project.translate.rust-lowering-unsupported]`: `subgroup-size`
- `project.translate.rust-refused: RustRefusal: type: glam has no 2x3 matrix of f32 (Mat2x3<f32>); rust-gpu shaders have square f32 and f64 matrices`: `matrix-resize`

</details>

<details>
<summary>2 untested: function 2</summary>

- `untested in the wgsl input row: no probe for parameters of type __atomic_compare_exchange_result<T>`: `atomicCompareExchangeWeak`
- `untested in the wgsl input row: the probe itself is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `subgroupElect`

</details>
