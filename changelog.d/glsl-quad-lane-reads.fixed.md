GLSL output of `QuadReadLaneAt` with a lane computed at run time, from HLSL, Slang or
CrossGL, calls a `crossglQuadReadLaneAt` helper that broadcasts each of the four quad
lanes with a constant index and selects one. It used to pass the lane to
`subgroupQuadBroadcast`, whose index GLSL requires to be a constant expression, so
glslang rejected the shader.
