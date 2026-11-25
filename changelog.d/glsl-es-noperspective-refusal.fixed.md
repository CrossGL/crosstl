GLSL ES and WebGL 2 output refuse a `noperspective` interpolation qualifier with
a typed `OpenGLLoweringRefused` or `WebGLLoweringRefused`, unless the program
enables `GL_NV_shader_noperspective_interpolation`. GLSL ES output used to write
it under `#version 300 es`, where glslang rejected `noperspective` as a reserved
word (#5060).
