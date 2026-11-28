# CrossTL translation matrix: Rust to WebGL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native Rust probe whose feature WebGL has too straight to WebGL, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 55.6 %: 10 supported, 7 refused, 1 wrong, 0 untested.

## Gaps

<details>
<summary>8 gaps: 7 refused, 1 wrong</summary>

Wrong, by cause:

- `the translation loses BuiltIn.InstanceIndex`: `instance-index`

Refused, by diagnostic:

- `project.translate.webgl-lowering-unsupported: WebGLLoweringRefused: WebGL cannot lower N construct: WebGL target does not support storage buffer resource '<probe>' (StructuredBuffer) (target webgl) [project.translate.webgl-lowering-unsupported]` (4): `Derivative::dfdx`, `Derivative::dfdy`, `Derivative::fwidth`, `SampledImage::sample`
- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @gl_FragDepth on parameter 'depth' of '<probe>' is not a fragment input in GLSL; GLSL declares gl_FragDepth as an output of fragment (GLSL N.N section N.N) (target webgl) [project.translate.builtin-value-unsupported]`: `fragment-depth`
- `project.translate.builtin-value-unsupported: BuiltinValueUnsupportedError: @gl_PointSize on parameter 'size' of '<probe>' is not a vertex input in GLSL; GLSL declares gl_PointSize as an input of fragment, geometry, tessellation_control, tessellation_evaluation and an output of geometry, mesh, tessellation_control, tessellation_evaluation, vertex (GLSL N.N section N.N) (targe`: `point-size`
- `project.translate.webgl-lowering-unsupported: WebGLLoweringRefused: WebGL cannot lower N construct: call to 'discard', which is neither a GLSL built-in function nor a function the program declares (target webgl) [project.translate.webgl-lowering-unsupported]`: `discard`

</details>
