Rust output lowers CrossGL's DXR inline ray queries to rust-gpu's `spirv_std::ray_tracing::RayQuery` instead of
refusing them. A `RayQuery<FLAGS>` local (or the GLSL importer's `rayQueryEXT`) is declared with `ray_query!`, and
`TraceRayInline`, `Proceed`, `Abort`, `CommitNonOpaqueTriangleHit` and the candidate and committed getters call the
spirv-std method that issues the same SPV_KHR_ray_query instruction; the query type's flags are combined with the
`initialize` flags as DXR combines them, and the ray flag and intersection kind constants are their values. An
acceleration structure is a `&AccelerationStructure` binding, `RayDesc` a struct with DXR's fields, and
`CommitProceduralPrimitiveHit` and the committed triangle's vertex positions, which spirv-std has no method for, are
inline SPIR-V. The object-to-world and world-to-object matrices (glam has no 4x3 or 3x4 matrix) and queries that are
parameters, struct members or module-scope variables are refused by name.
