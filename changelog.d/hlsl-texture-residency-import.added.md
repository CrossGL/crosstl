HLSL input imports the `out uint Status` overloads of the texture, storage-texture and
typed-buffer reads as CrossGL `textureResidency` queries and `CheckAccessFullyMapped` as
`textureResident`; both used to be refused. HLSL output writes a residency code as the read's
own Status, so `textureResident` becomes `CheckAccessFullyMapped` again. A sampled texture's
`Load(location, x)` with a `uint` `x` now imports as the offset overload DXC resolves it to,
not as a Status read, and HLSL output writes `texelFetchOffset` on multisample textures as
`Texture2DMS[Array].Load` instead of refusing it.
