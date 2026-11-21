OpenGL GLSL and WebGL output read the multiview view index (`gl_ViewIndex`,
HLSL `SV_ViewID`) in fragment shaders instead of refusing it. OpenGL output reads
it through `CROSSGL_VIEW_INDEX`, which is `gl_ViewIndex` under `GL_EXT_multiview`
for Vulkan and OpenGL SPIR-V and `int(gl_ViewID_OVR)` under `GL_OVR_multiview2`
otherwise; WebGL output reads `int(gl_ViewID_OVR)`. Geometry, tessellation and
mesh shaders keep `gl_ViewIndex` under `GL_EXT_multiview`. A vertex shader that
reads the view index is still refused, because `GL_OVR_multiview` needs its view
count there (#3578).
