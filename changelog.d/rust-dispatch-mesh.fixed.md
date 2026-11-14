Rust output writes a task stage's `DispatchMesh(x, y, z)` as rust-gpu's
`spirv_std::arch::emit_mesh_tasks_ext(x, y, z)` and `DispatchMesh(x, y, z, payload)` as
`emit_mesh_tasks_ext_payload(x, y, z, &mut payload)`, and Mojo output refuses `DispatchMesh` with a
`mojo-lowering-unsupported` diagnostic, since Mojo has no task stage. Both used to call a generated helper that did
nothing, so the task shader launched no mesh workgroups (#4392).
