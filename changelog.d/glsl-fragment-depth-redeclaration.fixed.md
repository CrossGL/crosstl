OpenGL output writes a `gl_FragDepth` redeclared with a depth layout
(`layout(depth_greater) out float gl_FragDepth;`) under its built-in name. The
declaration kept the name, but the writes to it were escaped to
`crossgl_gl_FragDepth`, which glslang rejected as an undeclared identifier, in
CrossGL output and in the GLSL round trip.
