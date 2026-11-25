GLSL output translates 64-bit integer buffer and shared-memory atomics: it requires
`GL_NV_shader_atomic_int64`, raises the version to GLSL 4.40 where glslang declares those overloads, converts 32-bit
value operands to the 64-bit memory type and writes `atomicLoad` as `atomicOr` with a 64-bit zero. It used to write
the calls without the extension, which glslang rejects (Metal `atomic_max_explicit` on `atomic_ulong`, CUDA
`atomicMax` on `unsigned long long`, HLSL `InterlockedMax` on `uint64_t`).
