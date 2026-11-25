HLSL output reduces a signed or 16-bit integer `subgroupAnd`, `subgroupOr` or `subgroupXor`
(`simd_and`, `OpGroupNonUniformBitwise*`) through the `uint` or `uint64_t` overload of
`WaveActiveBit*`, the only ones DXC declares, instead of writing a call DXC rejects. A vector
`subgroupAllEqual` (`OpGroupNonUniformAllEqual`) in a `bool` context, including an `if`
condition and an `&&` or `||` operand, becomes `all(WaveActiveAllEqual(...))`, which compares
the whole value as GLSL and SPIR-V do; it used to be refused.
