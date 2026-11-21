Metal input imports Metal's ballot masks. `simd_ballot`, `simd_active_threads_mask`, `quad_ballot` and
`quad_active_threads_mask` import as CrossGL's `WaveActiveBallot`, and a `simd_vote` or `quad_vote` variable,
parameter, member or `auto` local becomes a `uvec4`. An explicit conversion of a mask to its integer, such as
`static_cast<simd_vote::vote_t>(simd_ballot(p))` or `uint(vote.operator unsigned long())`, calls a helper that joins
the ballot's two low words or takes the four bits of the caller's quad. The importer used to write
`(vote_t)WaveActiveBallot(p)`, which CrossGL refused as an undeclared function, typed `auto vote = simd_ballot(p)` as
`bool`, and copied `simd_active_threads_mask()`, `quad_ballot(p)` and `simd_vote` locals into CrossGL unchanged.
`any()` called on a ballot imports as `WaveActiveAnyTrue` or `QuadAny`, and on a stored mask as a comparison of its
integer with zero; a default-constructed mask starts at zero. `all()` is refused, because on Apple GPUs it is false
in a group with an inactive thread, unlike `simd_all` and `quad_all`, and so is a mask constructed from an integer.
