# CrossTL translation matrix: HIP to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HIP probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 96.3 %: 232 supported, 2 refused, 7 wrong, 2 untested.

## Gaps

<details>
<summary>9 gaps: 2 refused, 7 wrong</summary>

Wrong, by cause:

- `'rgba32f' : format requires readonly or writeonly memory qualifier` (3): `storage-image`, `storage-image-3d`, `storage-image-array`
- `'+' : wrong operand types: no operation '+' exists that takes a left-hand operand of type ' temp highp float' and a right operand of type ' const int' (or there is no acceptable conversion)`: `implicit-conversion`
- `'=' : cannot convert from ' global lowp int' to ' temp highp uint'`: `__popc`
- `'mod' : undeclared identifier`: `reserved-global-name`
- `'rgba8ui' : format requires readonly or writeonly memory qualifier`: `unorm-storage-image`

Refused, by diagnostic:

- `project.translate.opengl-fmod-unrepresentable: OpenGLFloatingRemainderError: OpenGL cannot preserve floating remainder for ('f32', 'float'): unsupported-profile`: `float-remainder`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'f16' to 'f16': half unsupported profile`: `half`

</details>

<details>
<summary>2 untested: type 2</summary>

- `the target's own probe is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`

</details>
