CrossGL can write three texture operations other languages have (#4796). A
depth comparison takes an optional level-of-detail bias as its last operand
(`textureCompare(t[, s], P, ref[, bias])`, and the `Offset`, `Proj`,
`ProjOffset`, `Clamp` and `OffsetClamp` forms), as HLSL's `SampleCmpBias`
does. A depth-compare gather takes an optional component from 0 to 3
(`textureGatherCompare(t[, s], P, ref[, comp])` and its offset forms), as
HLSL's `GatherCmpGreen`, `GatherCmpBlue` and `GatherCmpAlpha` do.
`vec2 textureSamplePosition(t, int sample)` returns a multisample texture's
sample position, which HLSL output writes as `GetSamplePosition`. A target
whose generator does not lower one of them refuses it with
`TextureOperandUnsupportedError`; a gather component of 0 is dropped, and one
that is not an integer constant from 0 to 3 is refused with
`TextureOperandError`. Generators declare support with
`register_texture_operand_target`.
