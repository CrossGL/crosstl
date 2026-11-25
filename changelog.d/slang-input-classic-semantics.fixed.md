Slang input writes the `POSITION`, `NORMAL`, `TANGENT` and `BINORMAL` semantics (and their index 0) as `Position`,
`Normal`, `Tangent` and `Binormal`, the names the HLSL importer writes, so a Slang vertex input takes the semantic's
fixed location as the same HLSL input does, and a geometry input array such as `line float4 position[2] : POSITION`
gets a GLSL location.
