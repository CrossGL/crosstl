SPIR-V output reads `gl_InstanceID` (and HLSL's `SV_InstanceID`) as
`InstanceIndex - BaseInstance` with the `DrawParameters` capability, as slangc
and DXC do, so it counts from zero in draws with a non-zero first instance like
every other target. It used to decorate it as a bare `InstanceIndex`. The SPIR-V
importer now reads `BuiltIn InstanceIndex` as `gl_InstanceIndex`, so a SPIR-V
round trip keeps the bare built-in (#4506).
