GLSL ES output now raises `#version 300 es` to `#version 310 es` when a
fragment stage uses `layout(early_fragment_tests) in;`, desktop GLSL raises to
`#version 420 core` (unless `GL_ARB_shader_image_load_store` is enabled), and
WebGL 2 refuses the qualifier cleanly (#5059).
