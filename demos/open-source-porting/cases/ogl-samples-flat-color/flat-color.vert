#version 330 core
#define POSITION	0

precision highp float;
precision highp int;
layout(std140, column_major) uniform;

uniform mat4 MVP;

layout(location = POSITION) in vec4 Position;

void main()
{
	gl_Position = MVP * Position;
}

