Metal output rebuilds CUDA and HIP unions as MSL unions when Metal gives every member the recorded size and alignment, and refuses the ones whose layout differs, such as a CUDA `float3` member.
