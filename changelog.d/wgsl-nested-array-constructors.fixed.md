WGSL output writes a constructor of an array of arrays, such as
`float[2][3](float[3](...), float[3](...))`, as nested WGSL array value
constructors (`array<array<f32, 3>, 2>(...)`); it used to copy the outer
constructor as written, which WGSL compilers reject. A workgroup dimension set
through `local_size_x_id = N` names the program's own specialization constant
of id `N` in `@workgroup_size` instead of declaring a second override of the
same id.
