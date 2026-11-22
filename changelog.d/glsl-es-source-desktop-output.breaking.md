The target now chooses the GLSL profile. The `opengl` backend writes desktop GLSL, also from a GLSL ES
source (WebGL 2 or OpenGL ES): a CrossGL `#version 100 es` or `300 es` line starts output at `#version 330
core`, and `310 es` or `320 es` at `#version 450 core`, which the generator raises as before. Desktop output
leaves out GLSL ES `#extension` lines, starting from the desktop version that has the extension's constructs,
and refuses an extension desktop GLSL has no form of, such as `GL_OES_EGL_image_external_essl3`. The new
`gles` target (also `opengles` and `opengl-es`) writes GLSL ES for OpenGL ES 3.x: it keeps a GLSL ES
`#version` line or starts at `#version 300 es`, leaves out a `GL_ARB_` line whose constructs GLSL ES has in
its core language, and refuses one GLSL ES has no form of. WebGL output is still GLSL ES 3.00 and follows the
same extension rule.
