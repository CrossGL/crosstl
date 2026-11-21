OpenGL output now compiles for SPIR-V as well (glslang `-V` and `-G`) when a
varying has no fixed location. Such a varying takes `CROSSGL_LOCATION(n)`,
which expands to `layout(location = n)` only under `GL_SPIRV` or `VULKAN`,
with the location the SPIR-V output gives the same value; OpenGL still links
it by name and the `#version` is unchanged. Before, glslang rejected the
SPIR-V compilation with "SPIR-V requires location for user input/output".
