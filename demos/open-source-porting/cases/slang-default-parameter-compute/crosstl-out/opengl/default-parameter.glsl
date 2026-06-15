#version 450 core
layout(std430, binding = 0) buffer outputBufferBuffer { int outputBuffer[]; };
int helper(int val, int a) {
    return (val + a);
}

int test(int val) {
    return (helper(val, 16) + helper(val, 256));
}

// Compute Shader
layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;
void main() {
    int inVal = int(gl_GlobalInvocationID.x);
    int outVal = test(inVal);
    outputBuffer[gl_GlobalInvocationID.x] = outVal;
}
