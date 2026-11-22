Mojo output lowers CrossGL's wave and subgroup operations: reductions, inclusive, exclusive, clustered and
partitioned scans, votes, ballots and their bit counts, lane reads, shuffles, `WaveActiveAllEqual` and
`WaveMatch` become helpers that loop over the active lanes of a MAX warp and read each lane with `shuffle_idx`
under the active mask. `subgroupBarrier` is MAX's `syncwarp` between two fences, the `subgroupMemoryBarrier`
family is a fence, and CUDA's `__match_any_sync` matches bit patterns within the caller's 32-lane CUDA warp.
Quad operations are still refused.
