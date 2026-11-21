Metal and GLSL output lower the HLSL methods of a CrossGL `ByteAddressBuffer` or
`RWByteAddressBuffer` (`Load` to `Load4`, `Store` to `Store4`, `GetDimensions`
and the `Interlocked*` methods) as they lower the `buffer_load`, `buffer_store`,
`buffer_dimensions` and `atomic*` calls the HLSL importer writes for them.
They used to copy the method calls onto the `uint` word buffer they declare,
which the Metal compiler and glslang reject (#3646).
