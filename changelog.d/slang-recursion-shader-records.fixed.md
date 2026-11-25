Slang output refuses a recursive call cycle, which slangc rejects for every GPU target (error E55201), with the
`RecursionUnsupportedError` SPIR-V, WGSL and Metal output raise, and writes a function whose self calls are tail calls
as a loop. Slang output declares a shader record buffer as a `[[vk::shader_record]] ConstantBuffer<T, L>` in the
record's std430, std140 or scalar layout, and Slang input reads the layout argument back. SPIR-V, GLSL and Slang output
refuse a shader record that a stage other than a ray tracing stage reads (VUID-StandaloneSpirv-ShaderRecordBufferKHR-07119).
