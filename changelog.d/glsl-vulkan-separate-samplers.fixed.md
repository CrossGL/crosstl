OpenGL GLSL output keeps separate samplers and runtime-sized texture arrays for
a Vulkan compilation. A separate texture (`texture2D`) that is only sampled
through module-scope samplers is declared as `texture2D` under `#ifdef VULKAN`
and as the combined `sampler2D` of OpenGL otherwise, each sampler is declared
under `#ifdef VULKAN` at its own binding, and each sampling call picks
`sampler2D(texture, sampler)` or the combined uniform through the
`CROSSGL_SAMPLED(vulkan, opengl)` macro. An unsized texture array that is only
indexed stays unsized under `#ifdef VULKAN`, with GL_EXT_nonuniform_qualifier
required there. A Vulkan compilation of the output (glslang `-V`) had lost the
source's `OpTypeSampler` and `RuntimeDescriptorArray`; the OpenGL branch is the
combined output CrossTL wrote before (#4316, #4313, #4314, #4312).
