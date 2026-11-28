# CrossTL translation matrix: GLSL to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 94.7 %: 216 supported, 11 refused, 1 wrong, 0 untested.

## Gaps

<details>
<summary>12 gaps: 11 refused, 1 wrong</summary>

Wrong, by cause:

- `use of unknown declaration 'float16_t'`: `half`

Refused, by diagnostic:

- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: function: call to 'equal', a built-in function of another shading language that Mojo does not define and the program does not declare (target mojo) [project.translate.mojo-lowering-unsupported]` (3): `bool-vector-logic`, `integer-vector-comparison`, `vector-comparison`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and` (3): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`
- `project.translate.rule-inexpressible: RuleInexpressibleError: the signed ++ used as a value cannot be rewritten with wrapping arithmetic for mojo, whose code generator passes inout parameters by value; write it as a separate statement (target mojo) [project.translate.rule-inexpressible]` (2): `condition-assignment`, `lvalue-increment`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: Pair[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-struct-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N][N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals,`: `shared-2d-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: vec4[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-vector-array`

</details>
