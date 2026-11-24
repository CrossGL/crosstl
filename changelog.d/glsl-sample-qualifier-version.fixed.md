GLSL and GLSL ES output raise the `#version` line for a `sample` interpolation
qualifier to GLSL 4.00 or GLSL ES 3.20, unless the program enables
`GL_ARB_gpu_shader5` or `GL_OES_shader_multisample_interpolation`, and WebGL 2
output refuses it. GLSL ES output used to keep `#version 300 es`, where glslang
rejected `sample` as a reserved word (#5061).
