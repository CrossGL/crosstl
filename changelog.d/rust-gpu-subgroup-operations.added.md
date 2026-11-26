Rust output writes CrossGL wave operations as rust-gpu's `spirv_std::arch` subgroup functions: reductions, inclusive
and exclusive scans and clustered reductions by operand type (`subgroup_f_add`, `subgroup_s_min`,
`subgroup_logical_xor`, `subgroup_clustered_u_max::<8, _>`, ...), votes, `subgroup_elect`, ballots and their bit
counts, `subgroup_broadcast_first`, `subgroup_shuffle` and its xor/up/down forms, and quad broadcasts and swaps.
`WaveGetLaneIndex()` and `WaveGetLaneCount()` bind the `subgroup_local_invocation_id` and `subgroup_size` built-ins,
also in helpers. Rotations and partitioned operations are lowered through `subgroup_shuffle`. `WaveMatch`, the
quad votes, HLSL's exclusive `WaveMultiPrefix*` forms and non-constant cluster sizes are refused by name.
