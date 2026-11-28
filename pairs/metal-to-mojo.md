# CrossTL translation matrix: Metal to Mojo

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Metal probe whose feature Mojo has too straight to Mojo, and mojo checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 71.1 %: 182 supported, 70 refused, 4 wrong, 0 untested.

## Gaps

<details>
<summary>74 gaps: 70 refused, 4 wrong</summary>

Wrong, by cause:

- `cannot implicitly convert 'Int32' value to 'Bool'`: `bool-bitwise`
- `cannot implicitly convert 'StaticTuple[Float32, Int(N)]' value to 'Pair'`: `zero-initializer`
- `invalid call to 'exchange': value passed to '<probe>' cannot be converted from 'Pointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]' to 'Pointer[Float32, MutAnyOrigin]'`: `shared-memory-in-helper`
- `use of unknown declaration 'float16'; did you mean 'Float16'?`: `half`

Refused, by diagnostic:

- `project.translate.mojo-refused: MojoRefusal: Mojo kernel argument outp: RWStructuredBuffer<vec3> uses the metal layout, which places its elements or members at other offsets than Mojo's natural layout (a N-component vector takes N bytes and a vector is aligned to its size)` (39): `abs`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `ceil`, `clamp`, `copysign`, `cos`, `cosh`, `divide`, `exp`, `exp2`, `fabs`, `floor`, `fma`, `frexp`, `ldexp`, `log`, `log10`, `log2`, `max`, `min`, `modf`, `nextafter`, `pow`, `powr`, `round`, `rsqrt`, `saturate`, `sin`, `sinh`, `sqrt`, `tan`, `tanh`, `trunc`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: function: call to 'atomicThreadFence', a built-in function of another shading language that Mojo does not define and the program does not declare (target mojo) [project.translate.mojo-lowering-unsupported]` (5): `atomic_fetch_add`, `atomic_fetch_max`, `atomic_fetch_min`, `atomic_load`, `atomic_store`
- `project.translate.mojo-refused: MojoRefusal: Mojo kernel argument outp: RWStructuredBuffer<ivec3> uses the metal layout, which places its elements or members at other offsets than Mojo's natural layout (a N-component vector takes N bytes and a vector is aligned to its size)` (5): `clz`, `ctz`, `mulhi`, `popcount`, `reverse_bits`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter groups has no GPU index source for semantic <feature>; a Mojo kernel's arguments are the buffers and uniform values the host passes` (2): `num-subgroups`, `num-workgroups`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (N: int) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become kerne` (2): `constant-array-size`, `constant-expression`
- `project.translate.mojo-refused: MojoRefusal: Unsupported atomicMax for Mojo codegen; scalar integer target required: uint64` (2): `atomic_max`, `atomic_max_explicit`
- `project.translate.mojo-refused: MojoRefusal: Unsupported atomicMin for Mojo codegen; scalar integer target required: uint64` (2): `atomic_min`, `atomic_min_explicit`
- `project.translate.mojo-lowering-unsupported: MojoLoweringRefused: Mojo cannot lower N construct: function: call to 'workgroupExecutionBarrier', a built-in function of another shading language that Mojo does not define and the program does not declare (target mojo) [project.translate.mojo-lowering-unsupported]`: `threadgroup_barrier`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter group has no GPU index source for semantic gl_SubgroupID; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lane has no GPU index source for semantic gl_SubgroupInvocationID; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-invocation-id`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter lidx has no GPU index source for semantic gl_LocalInvocationIndex; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `local-invocation-index`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter size has no GPU index source for semantic gl_WorkGroupSize; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `workgroup-size`
- `project.translate.mojo-refused: MojoRefusal: Mojo compute entry parameter width has no GPU index source for semantic gl_SubgroupSize; a Mojo kernel's arguments are the buffers and uniform values the host passes`: `subgroup-size`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (OFFSETS: vec2[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which be`: `constant-vector-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (ORIGIN: Pair) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become`: `constant-struct`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (SCALE: float) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become`: `global-constant`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (WEIGHTS: float[N]) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which b`: `constant-array`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (scale: float) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become`: `global-shadowing`
- `project.translate.mojo-refused: MojoRefusal: Mojo has no global variables, so a writable initialized variable declared outside a function (var: float) has no Mojo spelling; only uniform values, storage buffers of a runtime-sized array and HLSL structured and byte-address buffers, which become kernel arguments, private variables, which become k`: `reserved-global-name`
- `project.translate.mojo-refused: MojoRefusal: Mojo kernel argument outp: the hlsl_structured layout of a RWStructuredBuffer element holding a matrix or a packed vector (Packed) is not Mojo's`: `struct-buffer-packing`

</details>
