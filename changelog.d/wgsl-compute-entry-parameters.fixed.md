WGSL output declares the bound parameters of a CrossGL compute entry (`@group(g) @binding(b) var<storage> p:
array<T>`) as module-scope bindings and reads its plain-value parameters, such as an imported CUDA kernel's `int n`,
from a `<entry>_Args` uniform block at the first free binding. It wrote them as entry-point parameters, which naga
rejects, and copied a storage buffer passed to an `out` parameter into a function variable of runtime size.
