#version 450 core

//vertex shader
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;

out vec2 vTexCoord;

void main()
{
    vTexCoord = texCoord;
    gl_Position = vec4(position, 1.0);
}

in vec2 vTexCoord;
out vec4 fragColor;

// fragment shader 

float marblePattern(vec2 uv) {
    return 0.5 + 0.5 * sin(10.0 * uv.x + iTime);
}

void main()
{
    vec3 color = vec3(marblePattern(vTexCoord), 0.0, 0.0);
    fragColor = vec4(color, 1.0);
}