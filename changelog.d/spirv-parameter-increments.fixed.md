SPIR-V output increments and decrements a by-value scalar parameter (`x++`) and a member of an `inout` structure
parameter (`s.c++`, `++s.c`). It wrote `OpLoad` of the parameter's value for the first, which spirv-val rejects, and
refused the second.
