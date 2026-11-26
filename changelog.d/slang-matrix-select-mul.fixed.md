Slang output writes a product with a matrix `select(...)` operand, such as
`select(n, m, c) * v`, as `mul(...)`. It used to write `*`, which slangc rejects
for a matrix and a vector and reads as a per-component product for two
matrices (#5191).
