GLSL output now writes each scalar of an array constructor as the element
type's constructor, e.g. `mat2[2](mat2(1.0), mat2(2.0))` instead of
`mat2[2](1.0, 2.0)`. It used to copy the scalar through, where glslang
rejected the constructor with "not enough data provided for construction"
(#3636).
