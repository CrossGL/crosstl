# CrossTL translation matrix: Slang to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Slang probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 93.9 %: 216 supported, 11 refused, 3 wrong, 0 untested.

## Gaps

<details>
<summary>14 gaps: 11 refused, 3 wrong</summary>

Wrong, by cause:

- `cannot implicitly convert 'StaticTuple[Float32, Int(N)]' value to 'Pair'` (2): `constant-struct`, `struct-constructor`
- `cannot implicitly convert 'SIMD[.int32, N]' value to 'SIMD[.uint32, N]'`: `countbits`

Refused, by diagnostic:

- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and` (3): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: function: call to '<feature>', a built-in function of another shading language that Mojo does not define and the program does not declare (target mojo) [project.translate.mojo-lowering-unsupported]` (2): `divide`, `fabs`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lidx has no GPU index source for semantic gl_LocalInvocationIndex; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `local-invocation-index`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: Pair[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-struct-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N][N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals,`: `shared-2d-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: vec4[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-vector-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo resource array binding count must resolve to a compile-time integer, got 'N'`: `constant-array-size`
- `project.translate.workgroup-uniform-load-unsupported: WorkgroupUniformLoadError: workgroupUniformLoad: '<probe>' is not a shared variable declared at program or stage scope for mojo (target mojo) [project.translate.workgroup-uniform-load-unsupported]`: `workgroupUniformLoad`

</details>
