SPIR-V, WGSL and Slang output now lower a union's `uchar4` byte view (`union { uint2 val; uchar4 bytes[2]; }`)
to the bytes of its storage words, reading each `u8vec4` as one word and writing one lane or a whole vector into
it. Narrow-integer legalization keeps a union's members at their declared width, so such a union is no longer
refused as an `ivec4[2]` view that does not match its two words.
