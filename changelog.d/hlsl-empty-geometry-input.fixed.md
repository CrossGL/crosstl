HLSL output of a GLSL or SPIR-V geometry shader that reads no per-vertex input declares an empty input vertex
structure. It used to get an `int _crossgl_empty;` padding member without a semantic, which DXC rejects in an entry
signature ("Semantic must be defined for all parameters"), so `EmitVertex` and `EndPrimitive` probes did not compile.
