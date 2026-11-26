HLSL output of a CUDA, HIP or OpenCL file with several kernels keeps each kernel's parameters apart. With
`entry_point=` it now drops the other kernels' hoisted buffers and constant-buffer members before pruning, and the
whole file renames a later kernel's parameter that collides with an earlier one's to `<kernel>_<name>`, so two
kernels taking `int width` or a buffer named `part` no longer write one HLSL name twice, which DXC rejects.
