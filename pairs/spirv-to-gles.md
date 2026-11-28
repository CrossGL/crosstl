# CrossTL translation matrix: SPIR-V to GLSL ES

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native SPIR-V probe whose feature GLSL ES has too straight to GLSL ES, and glslangValidator checked the result. It is compile-only. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 93.0 %: 422 supported, 27 refused, 5 wrong, 7 untested.

## Gaps

<details>
<summary>32 gaps: 27 refused, 5 wrong</summary>

Wrong, by cause:

- `'rgba32f' : format requires readonly or writeonly memory qualifier` (3): `OpImageQuerySize`, `OpImageRead`, `OpImageWrite`
- `'subgroupQuadBroadcast' : no matching overloaded function found`: `OpGroupNonUniformQuadBroadcast`
- `'subgroupShuffle' : no matching overloaded function found`: `OpGroupNonUniformBroadcast`

Refused, by diagnostic:

- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: built-in variable '<feature>' requires GL_EXT_clip_cull_distance; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]` (22): `geometry`, `geometry-input-lines`, `geometry-input-lines-adjacency`, `geometry-input-triangles`, `geometry-input-triangles-adjacency`, `geometry-invocations`, `geometry-output-line-strip`, `geometry-output-triangle-strip`, `instance-index`, `invariant-position`, `point-size`, `tessellation-domain-isolines`, `tessellation-domain-quads`, `tessellation-evaluation`, `tessellation-level-input`, `tessellation-patch-input`, `tessellation-point-mode`, `tessellation-spacing-fractional-even`, `tessellation-spacing-fractional-odd`, `tessellation-winding-cw`, `vertex`, `vertex-index`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N construct: GLSL version requirement: the 'noperspective' qualifier requires GL_NV_shader_noperspective_interpolation; the target is #version N es (target opengl) [project.translate.opengl-lowering-unsupported]`: `noperspective-interpolation`
- `project.translate.opengl-lowering-unsupported: OpenGLLoweringRefused: OpenGL cannot lower N constructs: GLSL version requirement: type 'double' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec2' does not exist in GLSL ES; the target is #version N es; GLSL version requirement: type 'dvec3' does not exist in GLSL ES; the t`: `OpGroupNonUniformPartitionEXT`
- `project.translate.opengl-refused: OpenGLRefusal: OpenGL target profile '#version N es' does not support GL_ARB_gpu_shader_int64; select a desktop GLSL profile for N-bit integer values`: `integer-literal-limits`
- `project.translate.opengl-refused: OpenGLRefusal: OpenGL texture operation 'textureLod' requires a declared texture or image resource argument: FunctionCallNode(function=IdentifierNode(name=nonuniformEXT), arguments=N)`: `array-of-textures-nonuniform-index`
- `project.translate.opengl-scalar-conversion-unsupported: OpenGLScalarConversionError: OpenGL cannot preserve source conversion from 'half' to 'half': half unsupported profile`: `half`

</details>

<details>
<summary>7 untested: graphics 5, type 2</summary>

- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_EXT_clip_cull_distance` (2): `clip-distance`, `cull-distance`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_NV_fragment_shader_interlock` (2): `rasterizer-ordered-buffer`, `rasterizer-ordered-image`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit integer' : not supported with this profile: es`: `int64`
- `the target's own probe is rejected: ERROR: 0:10: '64-bit unsigned integer' : not supported with this profile: es`: `uint64`
- `the target's own probe is rejected: ERROR: 0:2: '#extension' : extension not supported: GL_OES_viewport_array`: `viewport-index`

</details>
