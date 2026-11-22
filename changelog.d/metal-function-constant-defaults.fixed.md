The Metal importer writes a function constant with a default value,
`constant T x = is_function_constant_defined(raw) ? raw : D;` over `constant T raw [[function_constant(N)]];`, as
CrossGL's specialization constant `const T x @constant_id(N) = D;`. It used to keep the call to
`is_function_constant_defined`, which no target declares, so the translation was refused.
