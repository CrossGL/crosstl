Metal `precise::pow` keeps its 16 ulp bound on every target. Its finite path
called the target's own `pow`, whose accuracy GLSL, HLSL and SPIR-V inherit
from `exp2(y * log2(x))` and which Metal's fast-math mode replaces with
`fast::pow`. Results between 2^-120 and 2^120 were off by up to 483 ulp, and
results near 2^128 or 2^-126 overflowed or vanished. The finite path now runs
the fdlibm binary32 power over every normal result and overflow; only subnormal
results and subnormal operands keep the target's `pow`, since flushing them is
the target's choice. An unqualified `metal::pow` is unchanged. The new
conformance case `m_precise_pow` is within 1 ulp on Metal, OpenGL (llvmpipe)
and Vulkan (lavapipe, MoltenVK, KosmicKrisp) under strict and relaxed
compilation.
