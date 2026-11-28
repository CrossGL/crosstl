# CrossTL translation matrix: GLSL to HLSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native GLSL probe whose feature HLSL has too straight to HLSL, and dxc checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 97.4 %: 419 supported, 10 refused, 1 wrong, 0 untested.

## Gaps

<details>
<summary>11 gaps: 10 refused, 1 wrong</summary>

Wrong, by cause:

- `no matching function for call to 'grow'`: `buffer-element-argument`

Refused, by diagnostic:

- `project.translate.directx-unsized-array-unsupported: UnsizedArrayUnsupportedError: unsized array 'position' is declared at module scope: only a storage buffer's last member or an array of resources may omit its element count (HLSL allows unsized arrays only for resources, MSL array bounds are constant expressions, and GLSL and WGSL allow runtime-sized arrays only as the last membe` (7): `tessellation-control`, `tessellation-domain-isolines`, `tessellation-domain-quads`, `tessellation-evaluation`, `tessellation-level-input`, `tessellation-patch-input`, `tessellation-patch-output`
- `project.translate.directx-lowering-unsupported: DirectXLoweringRefused: DirectX cannot lower N constructs: function: call to 'beginInvocationInterlockARB', a built-in function of another shading language that DirectX does not define and the program does not declare; function: call to 'endInvocationInterlockARB', a built-in function of another shading language that Direc` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `project.translate.directx-lowering-unsupported: DirectXLoweringRefused: DirectX cannot lower N construct: function: call to 'interpolate_at_sample', a built-in function of another shading language that DirectX does not define and the program does not declare (target directx) [project.translate.directx-lowering-unsupported]`: `interpolateAtSample`

</details>
