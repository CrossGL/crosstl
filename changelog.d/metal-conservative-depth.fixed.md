Metal output writes a CrossGL conservative depth layout (`@depth_greater`,
`@depth_less`, `@depth_any`, `@depth_unchanged`) as the argument of the fragment
output's `[[depth(...)]]` attribute. A stage-level `gl_FragDepth` with a layout
used to get `[[attribute(n)]]`, which Metal rejects on a fragment output, and an
output structure member or output parameter got `[[depth(any)]]`, which drops the
promise. Metal import now keeps `[[depth(greater)]]` and `[[depth(less)]]` as
`@gl_FragDepth @depth_greater` and `@gl_FragDepth @depth_less`, so HLSL output
writes `SV_DepthGreaterEqual` and `SV_DepthLessEqual` for them. MSL has no
argument for `depth_unchanged`, so it is written as `depth(any)`, the same value
without the early depth test.
