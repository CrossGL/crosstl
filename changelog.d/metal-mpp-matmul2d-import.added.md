Metal input lowers MetalPerformancePrimitives `matmul2d` operations on
cooperative tensors with 16 or 32 element sizes, so kernels that use them
translate to every output language. Operations or layouts it cannot prove are
refused with `MetalMatmul2dError`. DirectX and OpenGL software subgroups now
answer lane-index queries in any function (#4400).
