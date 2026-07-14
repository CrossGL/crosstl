#version 450
layout(constant_id = 0) const uint BUFFER_ELEMENTS = 32;

layout(std430, binding = 0) buffer Pos {
    uint values[];
} pos;
// Compute Shader
uint fibonacci(uint n);

uint fibonacci(uint n) {
    if ((n <= 1)) {
        return n;
    }
    uint curr = 1;
    uint prev = 1;
    for (uint i = 2; (i < n); (++i)) {
        uint temp = curr;
        curr += prev;
        prev = temp;
    }
    return curr;
}

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
    uint index = gl_GlobalInvocationID.x;
    if ((index >= BUFFER_ELEMENTS)) {
        return;
    }
    pos.values[index] = fibonacci(pos.values[index]);
}
