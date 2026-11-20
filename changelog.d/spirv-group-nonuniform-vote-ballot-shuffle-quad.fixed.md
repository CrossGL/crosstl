The SPIR-V importer now reads the result of the vote, ballot, shuffle and quad
instructions instead of dropping it: `OpGroupNonUniformAll`, `Any` and
`AllEqual` become `subgroupAll`, `subgroupAny` and `subgroupAllEqual`,
`Elect` becomes `subgroupElect`, `Ballot` and `InverseBallot` become
`subgroupBallot` and `subgroupInverseBallot`, `BallotBitExtract`,
`BallotBitCount` (including the inclusive and exclusive scans),
`BallotFindLSB` and `BallotFindMSB` become their `subgroupBallot*` calls,
`Shuffle`, `ShuffleXor`, `ShuffleUp` and `ShuffleDown` become their
`subgroupShuffle*` calls, and `QuadBroadcast` and `QuadSwap` become
`subgroupQuadBroadcast` and the `subgroupQuadSwapHorizontal`,
`subgroupQuadSwapVertical` or `subgroupQuadSwapDiagonal` call for the
direction. The imported CrossGL used to read an undeclared `value_<id>`, so
translating it to any target failed. A `QuadSwap` whose direction is not the
constant 0, 1 or 2, and a `BallotBitCount` with an unknown group operation, are
refused with a diagnostic that names the instruction (#4570).
