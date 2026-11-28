WGSL output now writes a dereference of a storage buffer, `*buf` or
`*(buf + i)`, as the element access `buf[0]` or `buf[i]`, and refuses a
dereference of a buffer that has no element-access form. Previously, it copied
the `*` into the shader, which naga rejects because a WGSL storage array is not
a pointer (#4751).
