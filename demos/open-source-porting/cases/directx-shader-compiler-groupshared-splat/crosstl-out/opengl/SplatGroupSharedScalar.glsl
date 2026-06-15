#version 450 core
shared int a;
// Compute Shader
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
void main() {
    a = 123;
    ivec4 x = ivec4(a);
}
