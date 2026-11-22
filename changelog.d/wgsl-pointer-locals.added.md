WGSL output writes a CrossGL pointer local such as `float* p = &outp[i];` as
`let p = &outp[i];`, so a program that writes and reads through a pointer to a
buffer element, a local, workgroup or private variable, a structure member or a
matrix column translates instead of being refused. A pointer to a vector
component, re-pointing a pointer, pointer arithmetic or indexing, passing a
pointer on and writing a read-only buffer through one are refused by name,
because WGSL has no form for them.
