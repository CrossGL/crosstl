Metal output gives an explicit `@stage_entry` function named `main` its stage
name, such as `kernel_main`, `vertex_main` or `fragment_main`. It used to keep
`main`, which the Metal compiler rejects for kernel, vertex and fragment
functions (#4637).
