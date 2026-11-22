HLSL input and output carry the Shader Model 6.10 preview intrinsics `DebugBreak()`
and `TriangleObjectPositions()` (with its `BuiltInTrianglePositions` result), which
used to be refused as unknown calls. Every other target, Slang included, refuses them
with a typed diagnostic.
