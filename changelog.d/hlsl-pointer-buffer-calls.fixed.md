HLSL output writes `buffer_load` and `buffer_store` on a helper's `device T*` parameter as reads and writes through
the pointer's buffer and element offset (`data[off + i]`, `data[off + i] = v`). The store used to come out as
`data.Store(i, v)`, which DXC rejects on typed and structured buffers, so Metal matrix-buffer and row-major buffer
kernels did not compile as HLSL.
