HLSL output declares a CrossGL program-scope variable that has an initializer as
`static` with that initializer. It used to become a `$Globals` member, whose
initializer DXC ignores, so the shader read whatever the host bound. A brace list
assigned to a structure (a stage constant such as `const W S = {...};`) now goes
through a typed temporary that DXC accepts. HLSL input no longer carries the
initializer of an external global, which DXC ignores, into CrossGL.
