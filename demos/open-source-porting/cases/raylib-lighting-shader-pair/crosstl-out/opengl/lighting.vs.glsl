#version 330
layout(location = 0) in vec3 vertexPosition;
layout(location = 1) in vec2 vertexTexCoord;
layout(location = 2) in vec3 vertexNormal;
layout(location = 3) in vec4 vertexColor;
out vec3 fragPosition;
out vec2 fragTexCoord;
out vec4 fragColor;
out vec3 fragNormal;
// Constant Buffers
layout(std140) uniform Uniforms {
    mat4 mvp;
    mat4 matModel;
    mat4 matNormal;
};
// Vertex Shader
void main() {
    fragPosition = vec3((matModel * vec4(vertexPosition, 1.0)));
    fragTexCoord = vertexTexCoord;
    fragColor = vertexColor;
    fragNormal = normalize(vec3((matNormal * vec4(vertexNormal, 1.0))));
    gl_Position = (mvp * vec4(vertexPosition, 1.0));
    return;
}
