WGSL output writes barriers in helper functions that only compute entry points
reach, directly or through other helpers, which it used to refuse. Such a
helper is written in the compute stage's context, so its implicit-LOD texture
lookups also sample level 0 as the entry point's own lookups do. A barrier in
a helper that a vertex or fragment entry point also reaches is still refused,
because WGSL rejects it.
