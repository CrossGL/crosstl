Metal input imports an explicit conversion of a ballot mask to its integer, such as
`static_cast<simd_vote::vote_t>(simd_ballot(p))` or `quad_vote::vote_t(quad_active_threads_mask())`, through
CrossGL's `WaveActiveBallot` and a helper that joins the ballot's two low words or takes the four bits of the
caller's quad. The importer used to write `(vote_t)WaveActiveBallot(p)`, which CrossGL refused as an undeclared
function. `any()` called on a ballot imports as `WaveActiveAnyTrue` or `QuadAny`, where the importer used to write a
method call on a `uint4` that the Metal compiler rejected; `all()` called on a ballot is refused, because on Apple
GPUs it is false in a group with an inactive thread, unlike `simd_all` and `quad_all`.
