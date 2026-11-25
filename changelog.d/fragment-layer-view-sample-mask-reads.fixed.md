HLSL, Metal, Slang and WGSL output bind a GLSL fragment shader's reads of `gl_Layer`, `gl_ViewportIndex`,
`gl_ViewIndex` and `gl_SampleMaskIn[0]` as entry-point inputs (`SV_RenderTargetArrayIndex`, `SV_ViewportArrayIndex`,
`SV_ViewID`, `SV_Coverage` and their MSL attributes) and pass them to the helpers that read them; they were written
as undeclared names. A read of another `gl_SampleMaskIn` word is refused, because these targets deliver one 32-bit
coverage word, and WGSL refuses the three built-ins it lacks. A compute stage input declared as a stage variable
(`uint3 gl_GlobalInvocationID @input @gl_GlobalInvocationID;`, the SPIR-V and HIP import form) now reaches the
helpers that read it as a parameter.
