SPIR-V input imports `OpPtrAccessChain` and `OpInBoundsPtrAccessChain` by
adding their `Element` operand to the index of the array element the base
pointer names, so `&a[i]` moved by `e` reads `a[i + e]`. Before, `Element`
became one more subscript and the CrossGL no longer type-checked. A pointer
that does not name an array element, or whose `ArrayStride` differs from its
array's, is refused. A struct whose `OpName` is a built-in type name, such as
the `RWStructuredBuffer` block slangc writes, is declared as
`RWStructuredBuffer_` instead of shadowing the built-in type. SPIR-V output
accepts the GLSL declaration `int gl_SampleMask[1]` as a fragment output.
