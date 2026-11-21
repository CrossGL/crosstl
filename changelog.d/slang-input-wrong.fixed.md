Slang input keeps nested block and braced `case` scopes, the parentheses of nested prefix operators (`-(-x)`),
member functions, enumerator scopes (`E::A`), `packoffset` and `[[vk::offset]]` member placements,
`[earlydepthstencil]`, the conservative depth semantics and `[[vk::shader_record]]` constant buffers, and no longer
writes a prototype as an empty function. Each of these compiled differently after a Slang round trip. A generic member
function and a shader record that is not a constant buffer are refused. Slang output declares a texture passed to a
separately sampled parameter as a separate texture, and writes a component `packoffset` register.
