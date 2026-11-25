Metal output gives each entry its own argument table. A kernel takes its own OpenCL C or CUDA parameters, and a
CrossGL stage its own stage-block resources, and their bindings are checked in that entry's table, so two kernels of
one file may bind different buffers, or their `<kernel>_Args` blocks, at the same index (#3629, #3664). Metal output
used to refuse such a file, and when two kernels named different buffers alike it passed the first kernel's buffer
to both, at the first kernel's index. Stage functions of one name in two stages are now two functions; Metal output
used to define the name twice, which the Metal compiler rejects.
