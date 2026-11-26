OpenCL C input refuses a `printf` whose result is read (`int r = printf(...)`): OpenCL C's `printf` returns an `int`,
and CrossGL's `printf`, the Vulkan debug print, returns nothing, so the import wrote a value the targets cannot
compute. A `printf` statement is imported as before.
