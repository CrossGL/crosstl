Rust output selects vector components and matrix columns without glam's `Index`, `IndexMut` and `col_mut`, which
rust-gpu compiles to an `OpPhi` of Function-storage pointers that Vulkan's spirv-val rejects. A constant index names
the component or column (`v.y`, `m.z_axis`); a run-time index reads through spirv_std's `vector_extract_dynamic`,
`BVecN::test` or `Mat::col`, and a store, compound assignment or increment through it becomes a `match` on the index.
Bool vector components can now be assigned, and a constant index past the last component is refused.
