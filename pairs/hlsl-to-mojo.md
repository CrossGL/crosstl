# CrossTL translation matrix: HLSL to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HLSL probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 92.2 %: 200 supported, 13 refused, 4 wrong, 0 untested.

## Gaps

<details>
<summary>17 gaps: 13 refused, 4 wrong</summary>

Wrong, by cause:

- `cannot implicitly convert 'StaticTuple[Float32, Int(N)]' value to 'Pair'` (2): `constant-struct`, `struct-constructor`
- `cannot implicitly convert 'SIMD[.int32, N]' value to 'SIMD[.uint32, N]'`: `countbits`
- `use of unknown declaration 'float16'; did you mean 'Float16'?`: `half`

Refused, by diagnostic:

- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and` (3): `barrier-in-loop`, `shared-memory`, `shared-memory-in-helper`
- `project.translate.rule-inexpressible: RuleInexpressibleError: the signed ++ used as a value cannot be rewritten with wrapping arithmetic for mojo, whose code generator passes inout parameters by value; write it as a separate statement (target mojo) [project.translate.rule-inexpressible]` (2): `condition-assignment`, `lvalue-increment`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lidx has no GPU index source for semantic gl_LocalInvocationIndex; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `local-invocation-index`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no append or consume buffer: AppendStructuredBuffer obj needs a hidden element counter, and structured-buffer counters are not available in Mojo, whose kernels receive buffers as plain pointers`: `AppendStructuredBuffer.Append`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: Pair[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-struct-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: float[N][N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals,`: `shared-2d-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a variable declared outside a function (tile: vec4[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kernel locals, and`: `shared-vector-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo kernel argument outp: RWStructuredBuffer<Packed> uses the hlsl_structured layout, which places its elements or members at other offsets than Mojo's natural layout (a N-component vector takes N bytes and a vector is aligned to its size)`: `struct-buffer-packing`
- `project.translate.mojo-refused: MojoRefusal: Mojo resource array binding count must resolve to a compile-time integer, got 'N'`: `constant-array-size`
- `project.translate.mojo-refused: MojoRefusal: Unsupported Mojo buffer counter call buffer_increment_counter on RWStructuredBuffer obj; structured-buffer counters are not available in Mojo, whose kernels receive buffers as plain pointers`: `RWStructuredBuffer.IncrementCounter`

</details>
