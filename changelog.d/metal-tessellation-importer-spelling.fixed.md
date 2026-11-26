Metal import writes a `[[patch]]` vertex function in the HLSL importer's
spelling: `@domain("tri")` or `@domain("quad")` before the entry instead of
after its parameter list, where it was the return value's semantic, the
position in the patch as `@gl_TessCoord` instead of `@SV_DomainLocation`, which
GLSL has no form of, and no empty structure when the stage input held only the
control points. Metal output keeps the `gl_TessLevelOuter` and
`gl_TessLevelInner` members the HLSL importer writes for a domain shader's
patch constants out of the stage input, as it did `SV_TessFactor`, so HLSL
domain shaders translate to Metal again.
