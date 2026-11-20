Mojo output is written in the Mojo 1.2 language: functions are `def`, imports come from `std` and `max.gpu`,
structs use `@fieldwise_init`, fixed arrays are `Array`, and pointers name their origin. A compute stage's storage
buffers and uniform values become arguments of the kernel and of every function that reads them, because Mojo has
no global variables, and the GPU index built-ins are read from Mojo's `global_idx`, `thread_idx`, `block_idx`,
`block_dim` and `grid_dim`.
