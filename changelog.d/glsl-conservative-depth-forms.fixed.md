GLSL and WebGL output keep a conservative depth layout (`@depth_any`, `@depth_greater`,
`@depth_less`, `@depth_unchanged`) on every CrossGL form of the fragment depth output:
a stage-level `@output @gl_FragDepth` variable, an output structure member, an `out`
parameter and an entry's return value now redeclare `gl_FragDepth` with the layout.
The stage-level form used to declare an ordinary output with the layout, which glslang
rejects, and the others dropped it, so HLSL `SV_DepthGreaterEqual` and SPIR-V
`DepthGreater` lost their promise. GLSL ES output requires `GL_EXT_conservative_depth`,
desktop GLSL rises to 4.20, and a depth layout on any other output is refused.
