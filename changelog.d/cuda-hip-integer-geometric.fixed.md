CUDA and HIP output now converts integer scalar and vector operands of
`length()`, `distance()`, and `normalize()` to `float` and uses the standard
math or vector helper lowering. Previously, both backends copied those calls as
bare identifiers into the generated code, where CUDA and HIP compilers rejected
them (#3194).
