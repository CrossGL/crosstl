Every target except CrossGL, CUDA, HIP and WGSL binds a `var<storage>` variable that is not a runtime-sized
array as a storage buffer: a module-scope fixed-size array, scalar, vector or matrix becomes a buffer (`x[0]` for a
single value), a structure a std430 buffer block, and an entry parameter a runtime-sized buffer parameter. HLSL used
to place such a variable in the read-only `$Globals` constant buffer and GLSL in a `uniform`, Metal and WGSL copied
a fixed-size array parameter into a private array, and CUDA, HIP, Mojo and Rust passed a scalar parameter by value,
so stores never reached memory. A module-scope `var<uniform>` becomes a constant block with its group and binding,
and one with an initializer becomes a program constant; HLSL and GLSL used to drop the binding, and Metal and HLSL
dropped the initializer of an OpenCL `__constant` table. CUDA and HIP input import a namespace-scope `__device__` or
`__managed__` variable as a storage buffer and refuse one with an initializer; they used to import it as a
per-invocation private variable.
