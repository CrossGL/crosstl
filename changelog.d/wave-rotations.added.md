CrossGL has subgroup rotations, `WaveRotate(value, delta)` and
`WaveClusteredRotate(value, delta, clusterSize)`, with the semantics of SPIR-V
`OpGroupNonUniformRotateKHR`. GLSL input reads GL_KHR_shader_subgroup_rotate's
`subgroupRotate` and `subgroupClusteredRotate` as them, and
GL_EXT_shader_quad_control's `subgroupQuadAll` and `subgroupQuadAny` as `QuadAll`
and `QuadAny`. Every target reads the rotated lane with `WaveReadLaneAt` in a
helper function per operation and operand type, so HLSL, Metal, GLSL, SPIR-V,
CUDA, HIP, Slang and Rust output no longer refuse rotations.
