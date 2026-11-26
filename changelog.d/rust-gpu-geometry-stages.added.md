Rust output writes CrossGL geometry stages as rust-gpu `#[spirv(geometry(...))]` entry points instead of refusing
them. The stage layouts become the execution modes (`input_lines`, `output_triangle_strip`, `output_vertices = N`,
`invocations = K`), each module-scope per-vertex input array is an entry parameter taken by value and each output a
`&mut` reference the stage and its helpers write through, with the same locations as the other targets; the geometry
built-ins (`gl_InvocationID`, `gl_PrimitiveIDIn`, `gl_Position`, `gl_PointSize`, `gl_Layer`, `gl_ViewportIndex`,
`gl_PrimitiveID`) bind their rust-gpu built-ins. `EmitVertex`, `EndPrimitive`, `EmitStreamVertex(k)` and
`EndStreamPrimitive(k)` call `spirv_std::arch`. HLSL's stream-object form, inputs that are not one value per input
vertex, `noperspective`/`centroid`/`sample` values, outputs in streams other than 0 and non-constant stream numbers
are refused by name; the no-op stream placeholder types the generator used to write are gone.
