Metal input imports anonymous union and struct members of a class, such as the
`union { struct { half d; half m; }; half2 dm; }` of the ggml quantized blocks:
each is held by a member under a reserved name with the union layout contract,
and accesses and methods that name a promoted member reach it through that
member. Metal input used to refuse such a class. A `typedef struct { ... }` or
`typedef union { ... }` now also declares the unnamed aggregates its members
define; CrossGL output used to name them without declaring them.
