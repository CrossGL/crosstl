HLSL output converts a `T[N]` call result assigned to an array variable (`float a[3]; a = spread(x);`) to the array
through a typedef flat cast. The emitter holds such results as a vector or a `CrossGLFixedArray` struct, and
assigned them unchanged, which DXC rejects; SPIR-V imports declare every local this way.
