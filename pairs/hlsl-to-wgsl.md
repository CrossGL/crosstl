# CrossTL translation matrix: HLSL to WGSL

Measured on commit [`8f9a2e8`](https://github.com/CrossGL/crosstl/commit/8f9a2e88b1fda43915bc783f1c4a52557476f9a9) at 2026-11-28T09:51:00Z by the [Coverage workflow](https://github.com/CrossGL/crosstl/actions/workflows/coverage.yml). CrossTL translated each native HLSL probe whose feature WGSL has too straight to WGSL, and naga checked the result. Its compute probes also ran on a runtime. [COVERAGE.md](../COVERAGE.md#translation-matrix) describes the matrix and has every other cell.

Score 87.0 %: 394 supported, 54 refused, 5 wrong, 1 untested.

## Gaps

<details>
<summary>59 gaps: 54 refused, 5 wrong</summary>

Wrong, by cause:

- `the type of '<probe>' is expected to be 'vec2<u32>', but got 'vec2<i32>'` (3): `countbits`, `firstbithigh`, `firstbitlow`
- `the type of '<probe>' is expected to be 'vec2<i32>', but got 'vec2<f32>'`: `sign`
- `no matching call to 'textureSampleGrad(texture_1d<f32>, sampler, f32, f32, f32)'`: `Texture1D.SampleGrad`

Refused, by diagnostic:

- `project.translate.wgsl-refused: WGSLRefusal: WGSL target requires storage image resource img to declare a representable image format such as layout(rgba8) or @rgba8` (7): `integer-storage-image`, `read-write-image`, `storage-image`, `storage-image-1d`, `storage-image-3d`, `storage-image-array`, `unorm-storage-image`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type sampler1DArray yet; split texture/sampler/storage bindings are required` (6): `Texture1DArray.GetDimensions`, `Texture1DArray.Load`, `Texture1DArray.Sample`, `Texture1DArray.SampleBias`, `Texture1DArray.SampleGrad`, `Texture1DArray.SampleLevel`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL texture function textureCompareGrad yet` (6): `Texture1D.SampleCmpGrad`, `Texture1DArray.SampleCmpGrad`, `Texture2D.SampleCmpGrad`, `Texture2DArray.SampleCmpGrad`, `TextureCUBE.SampleCmpGrad`, `TextureCUBEArray.SampleCmpGrad`
- `project.translate.wgsl-texture-operand-unsupported: TextureOperandUnsupportedError: 'textureCompare' is a depth comparison with a level-of-detail bias, which wgsl output cannot carry: WGSL's textureSampleCompare takes no bias (WGSL section N.N) (target wgsl) [project.translate.wgsl-texture-operand-unsupported]` (6): `Texture1D.SampleCmpBias`, `Texture1DArray.SampleCmpBias`, `Texture2D.SampleCmpBias`, `Texture2DArray.SampleCmpBias`, `TextureCUBE.SampleCmpBias`, `TextureCUBEArray.SampleCmpBias`
- `project.translate.wgsl-foreign-builtin-call: WGSLForeignBuiltinCallError: WGSL target cannot lower the call to <feature>(): <feature> is a built-in function of another language, and WGSL declares no built-in function of that name (target wgsl) [project.translate.wgsl-foreign-builtin-call]` (5): `InterlockedCompareExchangeFloatBitwise`, `InterlockedCompareStore`, `InterlockedCompareStoreFloatBitwise`, `unpack_s8s16`, `unpack_u8u16`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target only lowers textureCompareLod() when the explicit LOD operand is literal N because textureSampleCompareLevel samples mip level N` (4): `Texture2D.SampleCmpLevel`, `Texture2DArray.SampleCmpLevel`, `TextureCUBE.SampleCmpLevel`, `TextureCUBEArray.SampleCmpLevel`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target requires textureCompareLod() to use a shadow/depth texture resource` (4): `Texture1D.SampleCmpLevel`, `Texture1D.SampleCmpLevelZero`, `Texture1DArray.SampleCmpLevel`, `Texture1DArray.SampleCmpLevelZero`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support storage buffer helper buffer_dimensions yet` (3): `RWStructuredBuffer.GetDimensions`, `StructuredBuffer.GetDimensions`, `runtime-array-length`
- `project.translate.wgsl-half-float-unsupported: HalfFloatUnsupportedError: the bit cast '<feature>' of a half value: WGSL f16 needs the optional shader-f16 feature (WGSL section N.N.N) (target wgsl) [project.translate.wgsl-half-float-unsupported]` (2): `asint16`, `asuint16`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support AppendStructuredBuffer resources yet` (2): `AppendStructuredBuffer.Append`, `AppendStructuredBuffer.GetDimensions`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target requires textureCompare() to use a shadow/depth texture resource` (2): `Texture1D.SampleCmp`, `Texture1DArray.SampleCmp`
- `project.translate.relative-vertex-id-unsupported: RelativeVertexIdUnsupportedError: gl_RelativeVertexID (Direct3D SV_VertexID): vertex entry point '<probe>' reads it, which excludes the draw's base vertex while the target's vertex index includes it, and WGSL has no base vertex built-in value to subtract; declare source_options={'zero_base_vertex': True} if every draw has a zero base v`: `vertex-index`
- `project.translate.wgsl-atomic-unsupported: WGSLAtomicLoweringError: WGSL target cannot lower this atomic access: outp is atomic f32 memory; WGSL has atomic<i32> and atomic<u32> only (WGSL section N.N.N)`: `InterlockedExchange`
- `project.translate.wgsl-foreign-builtin-call: WGSLForeignBuiltinCallError: WGSL target cannot lower the call to buffer_increment_counter(): buffer_increment_counter is a built-in function of another language, and WGSL declares no built-in function of that name (target wgsl) [project.translate.wgsl-foreign-builtin-call]`: `RWStructuredBuffer.IncrementCounter`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support ConsumeStructuredBuffer resources yet`: `ConsumeStructuredBuffer.GetDimensions`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type sampler2DMSArray yet; split texture/sampler/storage bindings are required`: `Texture2DArrayMS.Load`
- `project.translate.wgsl-refused: WGSLRefusal: WGSL target does not support CrossGL resource type samplerBuffer yet; split texture/sampler/storage bindings are required`: `Buffer.Load`
- `project.translate.wgsl-texture-bias-unsupported: WGSLTextureBiasUnsupportedError: WGSL target cannot lower texture() with a bias on obj: textureSampleBias has no overload for texture_1d<f32>; it takes 2D, 2D array, 3D, cube and cube array textures (target wgsl) [project.translate.wgsl-texture-bias-unsupported]`: `Texture1D.SampleBias`

</details>

<details>
<summary>1 untested: function 1</summary>

- `the target's own probe is rejected: error: no definition in scope for identifier: 'subgroupElect'`: `WaveIsFirstLane`

</details>
