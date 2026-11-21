Metal output renames a helper parameter or local that has the name of a buffer,
texture, sampler, uniform, constant-buffer member or threadgroup variable, instead
of refusing the program or emitting MSL in which the local hides the resource. A
CUDA helper `__device__ int twice(int o)` next to a kernel buffer `o`, a nested
local named like a buffer the helper also reads, and a threadgroup array whose
name a caller's local reuses now translate and compile. The renaming follows C
scope rules, so an initializer still reads the outer name.
