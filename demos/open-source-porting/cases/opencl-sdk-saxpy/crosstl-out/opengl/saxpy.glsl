
#version 450 core
layout(std430, binding = 0) buffer xBuffer { float x[]; };
layout(std430, binding = 1) buffer yBuffer { float y[]; };
// Constant Buffers
layout(std140, binding = 2) uniform saxpy_Args {
    float a;
};
// Compute Shader
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    int gid = int(gl_GlobalInvocationID.x);
    y[gid] = fma(a, x[gid], y[gid]);
}

