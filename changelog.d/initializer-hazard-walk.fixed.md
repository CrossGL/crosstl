Translating GLSL or CrossGL with deeply nested calls to HLSL, Metal, CUDA, HIP or
Slang no longer takes time exponential in the nesting depth. The check that
decides whether a local must be renamed for C++ declaration scoping visited each
call's arguments twice per level, so twelve nested `asint(asuint(...))` calls took
minutes; they now translate in about a second.
