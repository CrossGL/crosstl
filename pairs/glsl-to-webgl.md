# CrossTL translation matrix: GLSL to WebGL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL probe whose feature WebGL has too straight to WebGL, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 68.2 %: 15 supported, 7 refused, 0 wrong, 0 untested.

## Gaps

<details>
<summary>7 gaps: 7 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.webgl-lowering-unsupported: WebGLLoweringRefused: WebGL cannot lower N construct: WebGL target does not support GLSL buffer block resource 'b0' (target webgl) [project.translate.webgl-lowering-unsupported]` (7): `dFdx`, `dFdy`, `fwidth`, `texture`, `textureOffset`, `textureProj`, `textureProjOffset`

</details>
