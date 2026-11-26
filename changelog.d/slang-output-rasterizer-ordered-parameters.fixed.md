Slang output declares a function parameter that receives a `@rasterizer_ordered` resource, directly or through
another function, with the `RasterizerOrdered*` type, and writes an explicit `@rasterizer_ordered` parameter that
way instead of as a semantic. slangc rejected the `RW*` parameter the call passed the ordered view to (E30019). A
parameter that receives both an ordered and a plain resource is refused.
