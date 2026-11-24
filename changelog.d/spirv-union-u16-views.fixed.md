SPIR-V, WGSL and Slang output now lower a union's `ushort2` 16-bit lane view (`union { uint2 val; ushort2 halves[2]; }`)
to the lanes of its storage words, reading each `u16vec2` as one word and writing one lane or a whole vector into
it, with no `Int16` capability. Signed 16-bit lane views and `u16vec4` views stay refused, and other generators
keep refusing 16-bit lane views as before.
