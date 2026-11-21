Mojo output is written in the Mojo 1.2 language: functions are `def`, imports come from `std` and `max.gpu`,
structs are `@fieldwise_init` `TrivialRegisterPassable` values, fixed arrays are `StaticTuple`, and pointers name
their origin. A compute stage's storage buffers, `var<storage>` array parameters and uniform values become
arguments of the kernel and of every function that reads them, because Mojo has no global variables, and the GPU
index built-ins are read from Mojo's `global_idx`, `thread_idx`, `block_idx`, `block_dim` and `grid_dim`.
Workgroup-shared arrays are allocated in shared memory instead of per thread, `out` and `inout` parameters write
the caller's variable, `barrier()` and the memory barriers call Mojo's `barrier()` and `threadfence()` instead of
empty helpers, and arithmetic follows the C usual arithmetic conversions.
