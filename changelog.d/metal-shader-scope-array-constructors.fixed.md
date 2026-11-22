Metal output writes the array constructors in shader-scope initializers as
initializer lists (#3713, #3714, #3715): program-scope variables, stage-block
constants, members of a named structure constructor (`W{m: float[2](1.0, 2.0),
s: 3.0}`) and unsized constructors (`float[](1.0, 2.0)`). It used to write the
GLSL spelling, which the Metal compiler rejects. A constructor indexed with an
integer literal, `float[2](1.0, 2.0)[1]`, becomes the element it selects in every
output that builds arrays only in initializers; any other array constructor in
such an initializer is refused with a typed diagnostic.
