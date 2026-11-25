Rust output lowers CrossGL textures, samplers and storage images to rust-gpu `spirv_std::Image!` and `Sampler`
bindings instead of refusing them. Each image built-in calls the spirv-std method that issues the same SPIR-V
instruction (`sample`, `sample_by_lod`, `sample_by_gradient`, the projective and `sample_depth_reference` forms,
`gather`, `fetch_with_lod`, `query_size_lod`, `query_levels`, `query_samples`, `read`, `write`, `read_subpass`), and
1D-array images and texel buffers use inline SPIR-V. A combined sampler is split into an image and a sampler bound
beside it, and helpers that sample it receive both. A storage image's format becomes part of its `Image!` type, so a
helper parameter takes the format of the images its callers pass; loads from one-, two- and three-component formats
are widened to four components as GLSL defines. Samples outside fragment shaders read the base level, as for SPIR-V,
since rust-gpu rejects `ImplicitLod` instructions in compute and vertex entry points. Texel offsets, gathers with a
depth reference or a non-constant component, storage images and helper parameters without a format, a helper
parameter that receives images of two formats, multisampled storage reads and resource arrays of arrays are refused
by name. The Rust coverage probes declare `StorageImageExtendedFormats`, as glslang does for GLSL.
