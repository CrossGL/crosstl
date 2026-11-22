SPIR-V output of a tessellation evaluation stage that reads the patch constants as a structure, as HLSL domain
shaders and Metal post-tessellation vertex functions do, now validates. A narrower tessellation level member
(`float edges[3] : SV_TessFactor`, `float inside : SV_InsideTessFactor`) reads the leading elements of the
`TessLevelOuter`/`TessLevelInner` built-ins, which spirv-val used to reject as a mismatched constituent type; a
per-patch member is a `Patch` input placed after the control-point inputs, where SPIR-V output of the tessellation
control stage writes it, instead of a refused location conflict; and Metal's `[[position_in_patch]]` is the
`TessCoord` built-in.
