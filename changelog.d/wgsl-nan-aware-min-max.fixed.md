WGSL output translates the NaN-avoiding `fmin`, `fmax`, `fmin3` and `fmax3` that Metal, CUDA (`fminf`, `fmaxf`) and
HIP kernels call. WGSL declares none of them, so the generator refused those kernels; each call is now a helper per
operand type that tests both operands' binary32 bits and returns the one that is not a NaN, as SPIR-V's `NMin` and
`NMax` do. Integer or double operands stay refused.
