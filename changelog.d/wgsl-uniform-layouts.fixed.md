WGSL output reads a uniform block at the offsets its declared layout gives (std140 by default, std430 for push
constants, or `@buffer_layout`), where WGSL's own uniform layout differs: `mat2` and other two-row matrices,
row-major matrices, HLSL `cbuffer` packing and std430 scalar arrays. Such a block is declared as `array<vec4<T>, N>`
registers and each member is read from its register; the output used to declare the typed members, so it read
other bytes than the host wrote.
