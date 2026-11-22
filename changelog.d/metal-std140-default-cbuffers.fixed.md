Metal output lays out a CrossGL `cbuffer` that declares no `@buffer_layout` by
std140, as the GLSL and SPIR-V outputs do. It used to keep MSL's natural layout,
so a `vec3` took 16 bytes, a `bool` one, a `float` array had a 4-byte stride and
a `mat2` 8-byte columns, and the kernel read other bytes than the host wrote
(#3413, #3415, #3421). A cbuffer marked `@buffer_layout(metal)` keeps MSL's
layout, and std140 arrays of 64-bit integers are now declared as `long2` or
`long4` slots instead of being refused.
