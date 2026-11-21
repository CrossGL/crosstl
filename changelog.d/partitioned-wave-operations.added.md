CrossGL has partitioned wave reductions and scans: `WaveMultiSum`,
`WaveMultiProduct`, `WaveMultiMin`, `WaveMultiMax`, `WaveMultiBitAnd`,
`WaveMultiBitOr`, `WaveMultiBitXor`, the inclusive `WaveMultiPrefixInclusive*`
scans of each operator, and the exclusive `WaveMultiPrefixExclusiveMin` and
`WaveMultiPrefixExclusiveMax`. GLSL input reads every
GL_NV_shader_subgroup_partitioned function as one of them. Every target computes
them with a loop over the subgroup's lanes built from `WaveActiveBallot`,
`WaveReadLaneAt` and the lane index, so HLSL, Metal, GLSL, SPIR-V, CUDA, HIP,
Slang and Rust output no longer refuse them.
