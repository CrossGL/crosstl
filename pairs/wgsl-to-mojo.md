# CrossTL translation matrix: WGSL to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 92.6 %: 200 supported, 13 refused, 3 wrong, 2 untested.

## Gaps

<details>
<summary>16 gaps: 13 refused, 3 wrong</summary>

Wrong, by cause:

- `invalid call to '__lshift__': value passed to 'rhs' cannot be converted from 'SIMD[.uint32, N]' to 'SIMD[.int32, N]'`: `vector-shift`
- `invalid call to '__rshift__': value passed to 'rhs' cannot be converted from 'SIMD[.uint32, N]' to 'SIMD[.int32, N]'`: `integer-vector-arithmetic`
- `use of unknown declaration 'float'`: `array-of-arrays`

Refused, by diagnostic:

- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and` (3): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter groups has no GPU index source for semantic <feature>; a Mojo kernel's arguments are the buffers and uniform values the host passes` (2): `num-subgroups`, `num-workgroups`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter group has no GPU index source for semantic gl_SubgroupID; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lane has no GPU index source for semantic gl_SubgroupInvocationID; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-invocation-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lidx has no GPU index source for semantic gl_LocalInvocationIndex; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `local-invocation-index`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter width has no GPU index source for semantic gl_SubgroupSize; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-size`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (a0: float) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and cons`: `workgroupUniformLoad`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: Pair[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-struct-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N][N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals,`: `shared-2d-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: vec4[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-vector-array`

</details>

<details>
<summary>2 untested: function 2</summary>

- `untested in the wgsl input row: no probe for parameters of type __frexp_result_vec<N, T>`: `frexp`
- `untested in the wgsl input row: no probe for parameters of type __modf_result_vec<N, T>`: `modf`

</details>
