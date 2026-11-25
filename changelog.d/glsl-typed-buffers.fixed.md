GLSL output writes a CrossGL `Buffer<T>`, `RWBuffer<T>` or `RasterizerOrderedBuffer<T>` as a `samplerBuffer` or
`imageBuffer` texel buffer, with element reads, stores and atomics as texel-buffer calls, instead of keeping
`uniform Buffer<float>`, which glslang rejects (#3642). A storage image that a program function updates atomically
now gets the single-channel format image atomics need, so HLSL helpers that take an `RWBuffer` or `RWTexture2D`
and call `InterlockedAdd` on it compile.
