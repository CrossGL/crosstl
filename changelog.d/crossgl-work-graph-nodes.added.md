CrossGL has a `node` stage for D3D12 work-graph node shaders (#4794). A
`node { ... }` block holds one node: `@node_launch`, `@node_dispatch_grid`,
`@node_max_dispatch_grid`, `@node_max_recursion_depth`, `@node_id`,
`@node_share_input_of`, `@node_is_program_entry` and
`@node_local_root_arguments_table_index` on the entry; HLSL's input record,
output and output-records types with `@max_records`,
`@max_records_shared_with`, `@node_array_size`, `@unbounded_sparse_nodes`,
`@allow_sparse_nodes` and `@globallycoherent` on its parameters;
`@node_track_rw_input_sharing` and `@ SV_DispatchGrid` on records;
`GetRemainingRecursionLevels()` and `Barrier` with the Shader Model 6.8 flags.
The shared passes refuse a node that breaks a rule DXC enforces under
`lib_6_8` (launch mode, dispatch grid, thread group, input kind and count,
system values per launch mode, `globallycoherent`, cross-group sharing,
output-array attributes) with `WorkGraphNodeError`. Every target that has not
registered node shaders with `register_work_graph_target` refuses the stage
with `StageUnsupportedError`, and any other work-graph construct with
`WorkGraphUnsupportedError`. CrossGL output keeps nodes as written.
