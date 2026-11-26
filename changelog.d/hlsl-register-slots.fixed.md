HLSL output numbers the implicit `t`, `u` and `b` registers of buffers, textures, images and constant buffers in
one sequence, in declaration order, as SPIR-V, Vulkan GLSL and Metal number their bindings. Counting each class
from 0 put `readonly buffer fin; buffer outp;` at `t0` and `u0`, which DXC's SPIR-V back end binds both at binding
0. An implicit register now takes a slot that is free in all three classes, also of the registers the source gives
explicitly. Samplers keep their own `s` sequence.
