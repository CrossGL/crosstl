The metal-apple conformance harness binds the textures and samplers a Metal
compute kernel declares (`[[texture(n)]]`, `[[sampler(n)]]`, arrays of textures
and texture buffers): it creates deterministic float, half, int, uint, short and
ushort texels for 1D, 1D array, 2D, 2D array, 3D, cube, cube array and buffer
textures, and a nearest, repeating sampler with normalized coordinates, so the
coverage texture and image probes execute on the Apple GPU instead of ending as
harness errors. The Metal native texel-buffer and
array-of-textures-nonuniform-index probes convert their input through `int` as
the CrossGL probes do.
