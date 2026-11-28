# CrossTL translation matrix: SPIR-V to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native SPIR-V probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 75.0 %: 174 supported, 12 refused, 46 wrong, 2 untested.

## Gaps

<details>
<summary>58 gaps: 12 refused, 46 wrong</summary>

Wrong, by cause:

- `all members of 'TrivialRegisterPassable' struct must themselves be 'TrivialRegisterPassable'` (45): `Acos`, `Acosh`, `Asin`, `Asinh`, `Atan2`, `Atanh`, `Ceil`, `Cos`, `Cosh`, `Exp`, `Exp2`, `FAbs`, `FClamp`, `FMax`, `FMin`, `Floor`, `Fma`, `FrexpStruct`, `InverseSqrt`, `Ldexp`, `Log`, `Log2`, `ModfStruct`, `OpBitCount`, `OpBitReverse`, `OpGroupNonUniformFAdd`, `OpGroupNonUniformFMax`, `OpGroupNonUniformFMin`, `OpGroupNonUniformShuffle`, `OpGroupNonUniformShuffleDown`, `OpGroupNonUniformShuffleUp`, `OpGroupNonUniformShuffleXor`, `Pow`, `Round`, `Sin`, `Sinh`, `Sqrt`, `Tan`, `Tanh`, `Trunc`, `nested-struct-buffer`, `struct-array-buffer`, `struct-buffer`, `struct-buffer-copy`, `struct-buffer-packing`
- `cannot read from discard pattern '_'`: `buffer-in-helper`

Refused, by diagnostic:

- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and` (3): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (<feature>: uint) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become ker` (2): `local-invocation-index`, `subgroup-invocation-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (<feature>: uint) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel loca` (2): `num-subgroups`, `subgroup-size`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (gl_SubgroupID: uint) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals`: `subgroup-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (gl_NumWorkGroups: uint3) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel lo`: `num-workgroups`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: Pair[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-struct-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float4[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, an`: `shared-vector-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N][N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals,`: `shared-2d-array`

</details>

<details>
<summary>2 untested: type 2</summary>

- `slangc or spirv-dis is not installed` (2): `pointer-arithmetic`, `pointer-local`

</details>
