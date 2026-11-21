CUDA input reads an array declared with the `struct`, `union` or `class` keyword, such as `struct Pair ps[2];`, as
`array<Pair, 2>`, and `struct Pair* ptrs[2];` as an array of pointers. It used to write the undeclared type `Pair_2`
(#4698).
