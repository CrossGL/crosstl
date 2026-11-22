Slang output now writes a one-matrix constructor that changes the matrix's shape through a helper that keeps the
upper-left elements and fills the rest from the identity, swizzles a wider vector argument of a vector constructor to
the target's width, writes `b.length()` of a structured buffer as its element count, declares a scalar
`gl_CullDistance` member as a one-element array (slangc writes a scalar one as a non-array `CullDistance` built-in,
which the Vulkan specification forbids), and writes `MaybeReorderThread` as Slang's `ReorderThread`.
