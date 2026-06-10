
#version 450 core
out gl_PerVertex {
    vec4 gl_Position;
    mediump float gl_PointSize;
    float gl_ClipDistance[8];
    float gl_CullDistance[8];
};
layout(location = 0) in vec4 _ua_position;
layout(location = 0) out vec4 ANGLEXfbPosition;
// Vertex Shader
void main() {
    gl_Position = _ua_position;
    return;
}
