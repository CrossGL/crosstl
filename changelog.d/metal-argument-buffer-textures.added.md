The metal-apple conformance harness binds Metal argument buffers of textures and
samplers: a kernel argument that points to a structure whose members are all
textures or samplers gets four elements filled with the resource IDs of their
own textures and samplers. The Metal `unsized-array-of-textures` coverage
probes, which used to end as harness errors, run on the Apple GPU in both Metal
rows.
