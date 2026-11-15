CUDA and HIP input import `__shfl_up_sync`, `__shfl_down_sync` and the
cooperative-group `shfl_up`, `shfl_down` and `shfl_xor` of a tile as one
`WaveReadLaneAt` that runs in every lane, with only the source lane chosen by
the condition. The shuffle used to sit in one arm of `?:`, so it ran only in
some lanes and read lanes that were inactive there, which is undefined in HLSL,
SPIR-V, GLSL and CUDA (#4569).
