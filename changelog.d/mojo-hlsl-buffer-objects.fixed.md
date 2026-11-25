Mojo output passes an HLSL `StructuredBuffer`, `RWStructuredBuffer` or
`(RW)ByteAddressBuffer` declared outside a function to the kernel as an
`UnsafePointer` argument, reads and writes its elements or 32-bit words
through it, and answers `GetDimensions` from a `<name>_length` argument written
beside a buffer whose size the program queries. It used to declare the buffer
as a module-scope `var` of a placeholder type, which Mojo 1.2 rejects because it
has no global variables. Append and consume buffers, structured-buffer
counters, byte-address atomics, typed buffers, buffers passed to functions or
held in arrays or structures, and elements whose declared layout differs from
Mojo's (such as an HLSL `StructuredBuffer<float3>`) are refused instead.
