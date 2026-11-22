Slang output no longer types a wave operation's operands with the type its result is assigned to, so a `uint` lane
index inside `WaveReadLaneAt` on a `float4` value translates. A domain stage's `@partitioning` and `@outputtopology`
are validated and omitted, since Slang takes them from the hull, and `SV_DomainLocation` is `float2` on quad and
isoline domains.
