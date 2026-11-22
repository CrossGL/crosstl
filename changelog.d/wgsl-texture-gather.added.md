WGSL output writes texture gathers, level and sample-count queries, and
multisampled textures, which it used to refuse. `textureGather` and
`textureGatherOffset` become WGSL `textureGather(component, t, s, coords, ...)`
with the component as a constant from 0 to 3; a gather on a shadow sampler
with a reference, `textureGatherCompare` and `textureGatherCompareOffset`
become `textureGatherCompare`. `textureQueryLevels` and `textureSamples`
become `textureNumLevels` and `textureNumSamples`, converted to the `int` that
CrossGL returns. `sampler2DMS`, `isampler2DMS` and `usampler2DMS` declare a
`texture_multisampled_2d<T>` binding without a sampler and are read with
`texelFetch` and `textureSize`. Offset gathers on cube textures, gather
components that are not constants from 0 to 3, `textureGatherOffsets` and
multisampled array textures, which WGSL cannot express, are refused.
