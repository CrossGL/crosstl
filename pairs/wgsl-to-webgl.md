# CrossTL translation matrix: WGSL to WebGL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native WGSL probe whose feature WebGL has too straight to WebGL, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 80.0 %: 12 supported, 3 refused, 0 wrong, 0 untested.

## Gaps

<details>
<summary>3 gaps: 3 refused, 0 wrong</summary>

Refused, by diagnostic:

- `project.translate.webgl-lowering-unsupported: WebGLLoweringRefused: WebGL cannot lower N construct: WebGL target does not support storage buffer resource '<probe>' (RWStructuredBuffer) (target webgl) [project.translate.webgl-lowering-unsupported]` (3): `dpdx`, `dpdy`, `fwidth`

</details>
