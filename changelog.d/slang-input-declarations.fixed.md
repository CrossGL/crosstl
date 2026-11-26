Slang input imports more declarations: a `for` initializer that declares several variables (`int i = 0, n = 4`)
writes the type once, a C-style `typedef` of an array type is kept, a struct whose name is a CrossGL keyword is
renamed everywhere it is used, and a generic function keeps its generic parameters and their interface
constraints (`<T : IFloat & IArithmetic>`, `__generic<...>`, `let N : int`). Variadic (`each T`) and defaulted
generic parameters are refused by name. Slang output of a push-constant `ConstantBuffer<T>` whose `T` places its
members with `@offset` at std430-aligned offsets now writes them as `[[vk::offset]]` instead of refusing them;
`@align` is still refused.
