SPIR-V output lowers a negative integer texel offset such as `textureOffset(t, u, -1)` or `-ivec2(1, 0)` to `ConstOffset`. It used to refuse it as a non-constant offset (#4434).
