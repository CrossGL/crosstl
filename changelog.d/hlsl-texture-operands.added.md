HLSL input imports `SampleCmpBias` as CrossGL's biased `textureCompare[Offset][Clamp]`,
`GatherCmpGreen`, `GatherCmpBlue` and `GatherCmpAlpha` as `textureGatherCompare[Offset]`
with the component, and `GetSamplePosition` on `RWTexture2DMS[Array]` as
`textureSamplePosition`; each used to be refused. HLSL output writes the biased
comparisons as `SampleCmpBias` (Shader Model 6.8) and the component gathers as
`GatherCmp<Channel>`.
